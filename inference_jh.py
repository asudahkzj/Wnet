'''
Inference code for Wnet
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import torch.nn.functional as F
import json
import numpy as np

from evaluate.jaccard import db_eval_iou
from evaluate.f_boundary import db_eval_boundary
import scipy.io as scio
import csv

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--model_path', type=str, default="output/checkpoint.pth",
                        help="Path to the model weights.")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=36, type=int,
                        help="Number of frames")
    parser.add_argument('--num_ins', default=1, type=int,
                        help="Number of instances")
    parser.add_argument('--num_queries', default=36, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--kl_loss_coef', default=500, type=float)

    # dataset parameters
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--ann_path', type=str)
    parser.add_argument('--save_path', default='result.json')
    parser.add_argument('--dataset_file', default='jh')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    #parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
          [0.494, 0.000, 0.556], [0.494, 0.000, 0.000], [0.000, 0.745, 0.000],
          [0.700, 0.300, 0.600]]
transform = T.Compose([
    T.Resize(300),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b



def main(args):

    device = torch.device(args.device)
    # device = torch.device('cpu')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    num_frames = args.num_frames
    num_ins = args.num_ins
    with torch.no_grad():
        model, criterion, postprocessors = build_model(args)
        model.to(device)

        state_dict = torch.load(args.model_path)['model']
        model.load_state_dict(state_dict)

        with open('data/JHMDB/video_info.json', 'r') as fp:
            video_info = json.load(fp)

        ann_path = 'data/JHMDB/jhmdb_annotation.txt'
        missings = ["32_Pull-ups_at_193IBS_pullup_f_cm_np1_ba_med_0", 
                    "_Boom_Snap_Clap__challenge_clap_u_nm_np1_fr_med_1",
                    "Boom_Snap_Clap_clap_u_nm_np1_fr_med_1",
                    "_Boom_Snap_Clap__challenge_clap_u_nm_np1_fr_med_0",
                    "32_Pull-ups_at_193IBS_pullup_f_cm_np1_ba_med_1",
                    "Boom_Snap_Clap_clap_u_nm_np1_fr_med_0",
                    ]
        test_samples = []
        with open(ann_path, newline='') as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                video = row['video_id']
                if video in missings:
                    continue
                idx = np.linspace(4, len(video_info[video]['frames']) - 5, num=3,
                                        endpoint=True).astype(np.int32)
                idx = idx.astype(np.int64)
                # print(video, idx, len(self.video_info[video[:-4]]['frames']))
                for i in idx:
                    test_samples.append((video, i))
        
        iou = 0
        fb = 0
        print('Total num:', len(test_samples))
        for i in range(len(test_samples)):
            video_id, frame_idx = test_samples[i]
            frame_idx = int(frame_idx)
            class_ = video_info[video_id]['class']

            frame_path = os.path.join('data/JHMDB/Rename_Images/%s/%s' %
                                    (class_, video_id))
            need = []
            for j in os.listdir(frame_path):
                if j == '.AppleDouble' or j == '.DS_Store':
                    continue
                need.append(j)
            frames = list(map(lambda x: os.path.join(frame_path, x),
                            sorted(need, key=lambda x: int(x[:-4]))))

            step = 1
            all_frames = [i for i in range(frame_idx - 4 * step, frame_idx + 4 * step, step)]
            all_frames = []
            mid_frame = (args.num_frames-1)//2
            for j in range(args.num_frames):
                all_frames.append(frame_idx-mid_frame+j)
            for j in range(len(all_frames)):
                if all_frames[j] < 0:
                    all_frames[j] = 0
                elif all_frames[j] >= len(frames):
                    all_frames[j] = len(frames) - 1
            all_frames = np.asarray(frames)[all_frames]
            img_set = []
            for j in all_frames:
                im = Image.open(j)
                img_set.append(transform(im).unsqueeze(0).cuda())
            img=torch.cat(img_set,0)

            # audio
            a_filename = video_id+'.npy'
            audio = np.load(os.path.join('data/a2d_j_audio_feature', a_filename))
            audio = audio.transpose()
            audio = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)
            audio = audio.to(device)

            outputs = model(img, audio)
            masks = outputs['pred_masks'][0][mid_frame]
            pred_masks =F.interpolate(masks.reshape(1,num_ins,masks.shape[-2],masks.shape[-1]),(im.size[1],im.size[0]),mode="bilinear").sigmoid().cpu().detach().numpy()>0.5

            gt_path = 'data/JHMDB/puppet_mask/%s/%s/puppet_mask.mat' % (class_, video_id)
            fine_gt_mask = scio.loadmat(gt_path)['part_mask']
            try:
                assert fine_gt_mask.shape[2] == len(frames)
            except AssertionError:
                if frame_idx >= fine_gt_mask.shape[2]:
                    frame_idx = fine_gt_mask.shape[2] - 1
                print(class_, video_id, fine_gt_mask.shape, len(frames))
                # print(frames)
                # exit(1)
            fine_gt_mask = fine_gt_mask[:, :, frame_idx]
            
            single_iou = db_eval_iou(pred_masks[0][0], fine_gt_mask)
            single_fb, single_p, single_r = db_eval_boundary(pred_masks[0][0], fine_gt_mask)
            iou += single_iou
            fb += single_fb
            if i % 50 == 0:
                print(i+1, 'Jaccard:', iou / (i+1), ' F_boundary:', fb / (i+1))
        print('Total num:', len(test_samples))
        print('Jaccard:', iou/len(test_samples))
        print('F_boundary:', fb/len(test_samples))
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Wnet inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
