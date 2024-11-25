#!/usr/bin/env python
# coding: utf-8

# # Notes
# 
# We provide this notebook for inference and visualizations. 
# 
# You can either load images from a dataloader(see Sec. 1) or from a local path(see Sec. 2).
# 
# Welcome to join [IDEA](https://idea.edu.cn/en)([中文网址](https://idea.edu.cn/))!

# In[1]:


import os, sys
import torch, json
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
import cv2
from util import box_ops
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)
from misc.viz_utils import colorize, visualize_instances_dict
from skimage.segmentation import watershed
from misc.utils import get_bounding_box, remove_small_objects
import scipy.io as sio
import pickle
from util.box_ops import box_cxcywh_to_xyxy,box_xyxy_to_cxcywh,box_xywh_to_xyxy,generalized_box_iou
from PIL import Image
import datasets.transforms as T
from torchvision.ops import masks_to_boxes
from torch import nn
    
def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius: valid area around a point in setA to consider
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB

class SimpleMinsumMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        #bs, num_queries = outputs["pred_logits"].shape[:2]

        num_queries = outputs.shape[0]
        # Also concat the target labels and boxes
        #tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = targets
        out_bbox = outputs
        # Compute the classification cost.


        # Compute the L1 cost between boxes
        #print(out_bbox, tgt_bbox)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        #print(cost_bbox)
        # Compute the giou cost betwen boxes            
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)
        #print(cost_giou)
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.view(1,num_queries, -1)

        sizes = [tgt_bbox.shape[0]]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        #print(indices)
        return indices[0][0],indices[0][1],cost_giou



# In[6]:
img_path = '/mntnfs/med_data5/louwei/panuke/Fold3/samples/'
gt_path = '/mntnfs/med_data5/louwei/panuke/Fold3/labels/'
seg_mat_path = '/mntnfs/med_data5/louwei/swin_hovernet/base_pred/mat/'
detection_mat_path = '/mntnfs/med_lihaofeng/histopathology_datasets/DINO-HOVER2/Fold3/size1024/results/mat/'
save_patch_path = '/mntnfs/med_lihaofeng/histopathology_datasets/DINO-HOVER2/Fold3/size1024/results/crop_patches/'
matcher = SimpleMinsumMatcher(
            cost_class=1, cost_bbox=1, cost_giou=1,
            focal_alpha=0.25
        )    

# In[7]:

results = {}
files = os.listdir(img_path)

paired_all = []
unpaired_true_all = []
unpaired_pred_all = []
true_inst_type_all = []
pred_inst_type_all = []
img_indx = 0

for file in files:
    print(file)
    image_ori = Image.open(img_path + file).convert("RGB")
    image_ori = np.array(image_ori)
    image_visual = image_ori.copy()
    gt_inst_map = sio.loadmat(gt_path + file[:-4] + '.mat')['inst_map']
    
    
    seg_mat = sio.loadmat(seg_mat_path + file[:-4] + '.mat')
    det_mat = sio.loadmat(detection_mat_path + file[:-4] + '.mat')
    
    det_bbox = det_mat['bbox']
    #print(det_bbox.shape)
    if det_bbox.shape[0] == 0:
        det_bbox=torch.tensor(np.array([0,0,0,0])).unsqueeze(0)
    det_bbox = box_xywh_to_xyxy(torch.tensor(det_bbox,dtype=torch.float))/256
    det_bbox[det_bbox<0] = 0
    len_det = det_bbox.shape[0]
    seg_inst = seg_mat['inst_map']
    inst_map = torch.tensor(seg_inst)
    obj_ids = torch.unique(inst_map)
    obj_ids = obj_ids[1:]
    masks = inst_map == obj_ids[:, None, None]
    #print(obj_ids)
    seg_boxes = masks_to_boxes(masks)/256
    if seg_boxes.shape[0] == 0:
        seg_boxes=torch.tensor(np.array([0,0,0,0])).unsqueeze(0)
    len_seg = seg_boxes.shape[0]
    seg_boxes = seg_boxes.to(torch.float)
    #boxes = np.array(boxes)
    #seg_boxes = box_xyxy_to_cxcywh(boxes)
    #print(det_bbox.shape,seg_boxes.shape)
    index_i,index_j,cost_giou = matcher(det_bbox,seg_boxes)
    det_bbox = (det_bbox*256).cpu().numpy()
    seg_boxes = (seg_boxes*256).cpu().numpy()
    #print(cost_giou.shape) ##det_len * seg_len
    uncertain_bboxes = []
    certain_bboxes = []
    certain_ids = []
    for (i,j) in zip(index_i,index_j):
        #print(i,j)
        #print(det_bbox[i],seg_boxes[j],cost_giou[i][j])
        if -cost_giou[i][j] > 0.8:
            certain_bboxes.append(seg_boxes[j])
            certain_ids.append(j)
            #image_ori = cv2.rectangle(image_ori, (int(det_bbox[i][0]), int(det_bbox[i][1])), (int(det_bbox[i][2]), int(det_bbox[i][3])), (255,0,0), 2)
            #image_ori = cv2.rectangle(image_ori, (int(seg_boxes[j][0]), int(seg_boxes[j][1])), (int(seg_boxes[j][2]), int(seg_boxes[j][3])), (0,255,0),2)
        else:
            if ((det_bbox[i][2]-det_bbox[i][0])*(det_bbox[i][3]-det_bbox[i][1])) > ((seg_boxes[j][2]-seg_boxes[j][0])*(seg_boxes[j][3]-seg_boxes[j][1])):
                image_visual = cv2.rectangle(image_visual, (int(det_bbox[i][0]), int(det_bbox[i][1])), (int(det_bbox[i][2]), int(det_bbox[i][3])), (0,0,255), 2)
                image_visual = cv2.rectangle(image_visual, (int(seg_boxes[j][0]), int(seg_boxes[j][1])), (int(seg_boxes[j][2]), int(seg_boxes[j][3])), (0,255,0),2)
                uncertain_bboxes.append(det_bbox[i])
            else:
                image_visual = cv2.rectangle(image_visual, (int(det_bbox[i][0]), int(det_bbox[i][1])), (int(det_bbox[i][2]), int(det_bbox[i][3])), (255,0,0), 2)
                image_visual = cv2.rectangle(image_visual, (int(seg_boxes[j][0]), int(seg_boxes[j][1])), (int(seg_boxes[j][2]), int(seg_boxes[j][3])), (0,0,255),2)
                uncertain_bboxes.append(seg_boxes[j])

    
      
      
    
    if len_det >= len_seg:
        unpaired = [i for i in range(len_det) if i not in index_i]
        for i in unpaired:
            image_visual = cv2.rectangle(image_visual, (int(det_bbox[i][0]), int(det_bbox[i][1])), (int(det_bbox[i][2]), int(det_bbox[i][3])), (0,0,255), 2)
            uncertain_bboxes.append(det_bbox[i])
    else:
        unpaired = [i for i in range(len_seg) if i not in index_j]
        for i in unpaired:
            image_visual = cv2.rectangle(image_visual, (int(seg_boxes[i][0]), int(seg_boxes[i][1])), (int(seg_boxes[i][2]), int(seg_boxes[i][3])), (0,0,255), 2)
            uncertain_bboxes.append(seg_boxes[i])
    cv2.imwrite('visual/'+file,image_visual)
    num = 0
    for box in uncertain_bboxes:
        crop_image = image_ori[int(box[0]):int(box[2]+1),int(box[1]):int(box[3]+1),:]
        crop_inst = gt_inst_map[int(box[0]):int(box[2]+1),int(box[1]):int(box[3]+1)]
        crop_image = cv2.resize(crop_image.copy(), (image_ori.shape[0],image_ori.shape[1]), cv2.INTER_CUBIC)
        cv2.imwrite('crop_patches/'+file[:-4]+'_'+str(num)+'.png',crop_image)
        
        crop_inst = cv2.resize(crop_inst.copy(), (image_ori.shape[0],image_ori.shape[1]), cv2.INTER_NEAREST)
        #cv2.imwrite('crop_patches/'+file[:-4]+'_'+str(num)+'_inst.png',crop_inst*255)
        crop_inst = crop_inst[:,:,None]
        #print(crop_image.shape,crop_inst.shape)
        crop_img = np.concatenate([crop_image, crop_inst], axis=-1)
        np.save(save_patch_path + file[:-4]+'_'+str(num)+'.npy', crop_img)
        num += 1
    #print(index)
    