#!/usr/bin/env python
# coding: utf-8


import os, sys
import torch, json
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from main import build_model_main
from util.slconfig import SLConfig
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
from datasets import build_dataset
import cv2
import random
from torchvision.ops.boxes import nms
import warnings
from util import box_ops
import scipy
import matplotlib.pyplot as plt
from math import ceil
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)
from compute_stats import run_nuclei_inst_stat
from metrics.stats_utils import remap_label
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    optional_import,
)
from misc.viz_utils import colorize, visualize_instances_dict
from skimage.segmentation import watershed
from misc.utils import get_bounding_box, remove_small_objects
import scipy.io as sio
import pickle
from sahi.models.dino import DINOModel
from sahi.predict import get_sliced_prediction, predict, get_prediction

def process(pred_map,fuse_mask, nr_types=None, return_centroids=True):
    """Post processing script for image tiles.

    Args:
        pred_map: commbined output of tp, np and hv branches, in the same order
        nr_types: number of types considered at output of nc branch
        overlaid_img: img to overlay the predicted instances upon, `None` means no
        type_colour (dict) : `None` to use random, else overlay instances of a type to colour in the dict
        output_dtype: data type of output
    
    Returns:
        pred_inst:     pixel-wise nuclear instance segmentation prediction
        pred_type_out: pixel-wise nuclear type prediction 

    """
    if nr_types is not None:
        pred_type = pred_map[..., :1]
        pred_inst = pred_map[..., 1:]
        pred_type = pred_type.astype(np.int32)
    else:
        pred_inst = pred_map

    pred_inst = np.squeeze(pred_inst)
    pred_inst = __proc_np_hv(pred_inst,fuse_mask)

    inst_info_dict = None
    if return_centroids or nr_types is not None:
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background
        inst_info_dict = {}
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: chane format of bbox output
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            inst_map = inst_map[
                inst_bbox[0][0] : inst_bbox[1][0], inst_bbox[0][1] : inst_bbox[1][1]
            ]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            inst_contour = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            inst_contour = np.squeeze(inst_contour[0][0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sthg
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                continue # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)
            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y
            inst_info_dict[inst_id] = {  # inst_id should start at 1
                "bbox": inst_bbox,
                "centroid": inst_centroid,
                "contour": inst_contour,
                "type_prob": None,
                "type": None,
            }

    if nr_types is not None:
        #### * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    # print('here')
    # ! WARNING: ID MAY NOT BE CONTIGUOUS
    # inst_id in the dict maps to the same value in the `pred_inst`
    return pred_inst, inst_info_dict
def __proc_np_hv(pred,fuse_mask):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.45, dtype=np.int32)
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    ##marker dist 
    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)
    overall = np.array(overall >= 0.45, dtype=np.int32)  #0.45
    
    seg_marker = blb - overall
    seg_marker[seg_marker < 0] = 0
    seg_marker = binary_fill_holes(seg_marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seg_marker = cv2.morphologyEx(seg_marker, cv2.MORPH_OPEN, kernel)
    seg_marker = measurements.label(seg_marker)[0]
    seg_marker = remove_small_objects(seg_marker, min_size=10)
    marker = marker_fusion(seg_marker,fuse_mask)
    marker = fuse_mask
    proced_pred = watershed(dist, markers=marker, mask=blb)
    proced_pred = remove_small_objects(proced_pred, min_size=30)

    return proced_pred 
def marker_fusion(seg_marker,det_marker):
    inst_ids = np.unique(seg_marker)[1:]
    det_ids = np.unique(det_marker)
    max_id = det_ids[-1]
    mask = np.zeros((seg_marker.shape[0],seg_marker.shape[1]))
    for idx in inst_ids:
        mask[seg_marker==idx] = 1
        search = np.array(det_marker[seg_marker==idx])
        if search.max()==0:
            det_marker[seg_marker==idx] = max_id + 1
            max_id += 1
    return det_marker
    
def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)
    

    
def sliding_window_inference(
    inputs: torch.Tensor,
    image_visual,
    roi_size: Union[Sequence[int], int],
    pad_size,
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    postprocessors,
    det_score,
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.5,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 255.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    test_size = 1024
    transformer = T.Compose([T.RandomResize([test_size], max_size=test_size),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    compute_dtype = inputs.dtype
    max_h = inputs.shape[-1]
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")
    predictor.eval()
    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape
    process_device = 'cpu'
    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = [0,0,0,0]
    #for k in range(len(inputs.shape) - 1, 1, -1):
        #diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        #half = diff // 2
        #pad_size.extend([half, diff - half])
    #print(pad_size)
    
    #print('inputs',inputs.shape)
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)
    #print(scan_interval)
    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    #print(image_size, roi_size,len(slices))
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=process_device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(importance_map, torch.Tensor, process_device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    # Perform predictions
    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    is_tensor_output = True  # whether the predictor's output is a tensor (instead of dict/tuple)
    patch_idx = 0
    # for each patch
    all_bounding_boxes = []
    all_scores = []
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        #print(slice_range)
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]

        start_x1,start_y1,start_x2,start_y2 = slices[slice_range[0]][0].start,slices[slice_range[0]][1].start,slices[slice_range[0]][0].stop,slices[slice_range[0]][1].stop



        window_data = convert_data_type(inputs[:,:,start_x1:start_x2,start_y1:start_y2], torch.Tensor)[0]

        window_data = Image.fromarray(window_data.squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)).convert("RGB")
        window_data,_ = transformer(window_data,None)
        window_data = window_data.unsqueeze(0)
        pad_y1,pad_y2,pad_x1,pad_x2 = 128,128,128,128
        window_data = F.pad(window_data, pad=(pad_y1,pad_y2,pad_x1,pad_x2), mode=look_up_option(padding_mode, PytorchPadMode), value=cval)


        with torch.no_grad():
            outputs,seg_binary,seg_hv = predictor(window_data.to('cuda'))  # batched patch segmentation

            outputs = postprocessors['seg'](outputs, torch.Tensor([[1.0, 1.0]]).to('cuda'))[0]
        det_masks = outputs['masks'][:,:,32:-32,32:-32].to(process_device)

        scores = outputs['scores']
        select_mask = scores > det_score
        select_score = scores[select_mask]
        boxes = box_ops.box_xyxy_to_cxcywh(outputs['boxes'])
        bboxes = boxes[select_mask].to(process_device)
        W,H = seg_binary.shape[-1],seg_binary.shape[-2]
        #visual_box = np.zeros((W, H))
        for b,s in zip(bboxes,select_score):
            unnormbbox = b * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            if bbox_x >= 32 and bbox_y >=32:
                bbox_x = bbox_x + (start_y1)-32
                bbox_y = bbox_y + (start_x1)-32
                bbox_w = bbox_w
                bbox_h = bbox_h

                
                all_bounding_boxes.append((bbox_x,bbox_y,bbox_w,bbox_h))
                all_scores.append(s)

        mask = F.softmax(det_masks[select_mask], dim=1)[:,1,:,:]
        #print(mask.shape)
        patch_idx += 1
        if mask.shape[0] > 0:
            mask,_ = torch.max(mask,dim=0)

            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.zeros((1,1,H-64,W-64))

        seg_binary = seg_binary[:,:,32:-32,32:-32].to(process_device)
        seg_hv = seg_hv[:,:,32:-32,32:-32].to(process_device)
        seg_prob_out = [seg_binary,seg_hv,mask]
        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())  # track predictor's output keys
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        # for each output in multi-output list
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(process_device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scale: out_roi_size/in_roi_size
            zoom_scale = []
            for axis, img_s_i in enumerate(
                zip(image_size)
            ):
                _scale = 1.0
                zoom_scale.append(_scale)

            if _initialized_ss < ss:  # init. the ss-th buffer at the first iteration
                # construct multi-resolution outputs
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype))
                _initialized_ss += 1

            # resizing the importance_map
            resizer = Resize(spatial_size=seg_prob.shape[2:], mode="nearest", anti_aliasing=False)

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                original_idx_zoom = list(original_idx)  # 4D for 2D image, 5D for 3D image
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        warnings.warn(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. "
                            f"Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). "
                            f"Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. "
                            f"Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n"
                            f"Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. "
                            "Tips: if overlap*roi_size*zoom_scale is an integer, it usually works."
                        )
                    original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(compute_dtype)
                # store results and weights
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]

                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)
    # remove padding if image_size smaller than roi_size
    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn("Sliding window inference results contain NaN or Inf.")

        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore
    final_output = final_output[0] if is_tensor_output else final_output

    if isinstance(inputs, MetaTensor):
        final_output = convert_to_dst_type(final_output, inputs, device=process_device)[0]  # type: ignore
    return final_output,all_bounding_boxes,all_scores
# # 0. Initialize and Load Pre-trained Models

# In[2]:


model_config_path = "config/DINO/DINO_4scale_swin.py" # change the path of the model config file


# In[3]:


args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 




# In[4]:


# load coco names
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}



# # 2. Visualize Custom Images

# In[5]:


from PIL import Image
import datasets.transforms as T


# In[6]:
img_path = 'monusac/test/images/'
save_seg_path = 'monusac/results/overlay/'

save_mat_path = 'monusac/results/mat/'

gt_path = 'monusac/test/Labels/'

if not os.path.exists(save_seg_path):
    os.makedirs(save_seg_path)

if not os.path.exists(save_mat_path):
    os.makedirs(save_mat_path)
# In[7]:

test_size = 1000
patch_size = 256
roi = (patch_size, patch_size)
pad_size = (128,128,128,128)
overlap = 0.6
sw_batch_size = 1
det_score = 0.45

# transform images
transform = T.Compose([
    #T.RandomResize([test_size], max_size=test_size),
    T.ToTensor(),
    #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
results = {}

#detection_model = DINOModel(model=model,processor=postprocessors,mask_threshold=0.5,confidence_threshold=det_score,device=args.device,image_size=patch_size)

paired_all = []
unpaired_true_all = []
unpaired_pred_all = []
true_inst_type_all = []
pred_inst_type_all = []
img_indx = 0
def padding(img):

    diff_h = int(256) - int(img.shape[0])
    padt = max(0,diff_h // 2)
    padb = max(0,diff_h - padt)

    diff_w = int(256) - int(img.shape[1])
    padl = max(0,diff_w // 2)
    padr = max(0,diff_w - padl)
    img = np.pad(img, ((padt,padb),(padl,padr),(0,0)), 'constant')
    return img,padt,padb,padl,padr
model_checkpoint_path = 'checkpoint_monusac.pth' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.to('cuda')
_ = model.eval()
files = os.listdir(img_path)

img_indx = 0

for file in files:
    image_ori = Image.open(img_path + file).convert("RGB") # load image
    image_ori = np.array(image_ori)
    image_visual = np.array(image_ori)

    padt,padb,padl,padr = 0,0,0,0
    tag = False
    if image_ori.shape[1] < 256 or image_ori.shape[0] < 256:
        tag = True
        image_ori,padt,padb,padl,padr = padding(image_ori)
    image = torch.Tensor(image_ori).permute(2,0,1)

    with torch.no_grad():
        (binary_map,hv_map,det_masks),all_bounding_boxes,all_scores = sliding_window_inference(image.unsqueeze(0), image_ori.copy(),roi,pad_size, sw_batch_size, model,postprocessors,det_score, overlap=overlap,padding_mode='constant')
        det_masks = det_masks.squeeze(0).squeeze(0)

    pred_dict = {'np': binary_map, 'hv': hv_map}
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]  # NHWC
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:]
    pred_output = torch.cat(list(pred_dict.values()), -1).cpu().numpy() # NHW3
    #pred_map = np.squeeze(pred_output) # HW3
    #print(pred_map.shape)
    i = 1
    sort_masks = []
    W, H = image_ori.shape[0],image_ori.shape[1]
    
    select_boxes = []
    #image_before = image_visual.copy()
    for b in all_bounding_boxes:
        bbox_x, bbox_y, bbox_w, bbox_h = b
        select_boxes.append((bbox_x,bbox_y,bbox_x+bbox_w, bbox_y+bbox_h))
        #image_before = cv2.rectangle(image_before, (int(bbox_x), int(bbox_y)), (int(bbox_x+bbox_w), int(bbox_y+bbox_h)), (0,255,255), 2)
        i += 1
    select_boxes = torch.Tensor(select_boxes)
    all_scores =  torch.Tensor(all_scores)
    item_indices = nms(select_boxes, all_scores, iou_threshold=0.2)
    filter_boxes = [all_bounding_boxes[i] for i in item_indices.tolist()]
    i = 0
    filter_map = np.zeros((W, H))
    center = []
    for b in filter_boxes:
        bbox_x, bbox_y, bbox_w, bbox_h = b
        filter_map[int(bbox_y):int(bbox_y+bbox_h),int(bbox_x):int(bbox_x+bbox_w)]=i
        center.append((int(bbox_x+bbox_w/2),int(bbox_y+bbox_h/2)))
        i += 1    

    mask_dino = np.zeros((W, H))
    i = 1
    for p in center:
        mask_dino[(p[1]-1):(p[1]+1),(p[0]-1):(p[0]+1)] = i
        i += 1
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_dino = cv2.dilate(mask_dino, kernel, 2)
    mask_dino = remap_label(mask_dino, by_size=False)        

    filter_map = remap_label(filter_map, by_size=False)        

    det_masks = det_masks.unsqueeze(0)
    #print(det_masks)
    fuse_mask = np.max([det_masks.cpu().numpy(),pred_output[...,0]],axis=0).squeeze(0)

    pred_output[...,0] = fuse_mask
    

    if tag:
        h,w = pred_output.shape[1],pred_output.shape[2]
        pred_output = pred_output[:,padt:h-padb,padl:w-padr,:]
        mask_dino = mask_dino[padt:h-padb,padl:w-padr]

    pred_inst, inst_info_dict = process(pred_output,mask_dino)

    pred_inst = pred_inst.astype(np.int16)

    overlaid_img = visualize_instances_dict(
        image_visual, inst_info_dict
    )

    cv2.imwrite(save_seg_path + file, cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))
    

    pred_mat = {}
    pred_mat["inst_map"] = pred_inst

    sio.savemat(os.path.join(save_mat_path, file[:-4]+".mat"), pred_mat)

run_nuclei_inst_stat(save_mat_path,gt_path)
