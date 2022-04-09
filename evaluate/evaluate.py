from math import sin
import numpy as np
from PIL import Image
import json
import os
from pycocotools import mask as coco_mask
from jaccard import db_eval_iou
import torch
from f_boundary import db_eval_boundary
import sys

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = np.zeros((height,width))
            print('-----')
        else:
            # rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(polygons)
            # if len(mask.shape) < 3:
            #     mask = mask[..., None]
            # # mask = torch.as_tensor(mask, dtype=torch.uint8)
            # mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = np.stack(masks)
    else:
        masks = np.zeros((0, height, width))
    return masks


if __name__ == "__main__":
    result_path = sys.argv[1]
    print(result_path)
    with open(result_path) as f:
        results = json.load(f)
    folder = 'data/rvos/train/Annotations/'
    videos = json.load(open('data/rvos/ann/instances_test_sub.json','rb'))['videos']
    vis_num = len(videos)
    j = 0
    count = 0
    iou = 0
    fb = 0
    frame_iou_list, frame_fb_list = [], []
    for i in range(vis_num):
        video = videos[i]
        file_names = video['file_names']
        id = video['id']
        w, h = video['width'], video['height']
        masks = []
        for l in range(len(file_names)):
            mask_path = os.path.join(folder, file_names[l][:-3]+'png')
            mask = np.array(Image.open(mask_path))
            masks.append(mask)
        while j < len(results):
            result = results[j]
            vid = result['video_id']
            if vid != id:
                break 
            oid = result['obj_id']
            segmentations = result['segmentations']
            segs = convert_coco_poly_to_mask(segmentations, h ,w)
            j += 1
            for k in range(len(file_names)):
                assert len(segs) == len(file_names)
                seg = segs[k]
                mask = masks[k]
                msk = mask == oid
                frame_iou = db_eval_iou(seg, msk)
                frame_iou_list.append(frame_iou)

                frame_f ,frame_p, frame_r = db_eval_boundary(seg, msk)
                frame_fb_list.append(frame_f)
            count += 1
            if count % 50 == 0:
                print('Jaccard:', np.array(frame_iou_list).sum() / len(frame_iou_list), 
                      ' F_boundary:', np.array(frame_fb_list).sum() / len(frame_fb_list))

    print('Total num:', len(frame_fb_list))
    frame_num = len(frame_iou_list)
    frame_iou_list = np.array(frame_iou_list)
    print('Jaccard:', frame_iou_list.sum() / frame_num)
    frame_fb_list = np.array(frame_fb_list)
    print('F_boundary:', frame_fb_list.sum() / len(frame_fb_list))