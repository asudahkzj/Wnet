"""
YoutubeVIS data loader
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
import csv
import h5py

import json
import numpy as np

class AVOSDataset:
    def __init__(self, paths, img_folder, mask_folder, ann_file, exp_file, transforms, return_masks, num_frames):
        # ytvos
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.ann_file = ann_file
        self.exp_file = exp_file
        self.audio_ytvos = 'data/rvos_audio_feature'
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ytvos = YTVOS(ann_file)
        # self.cat_ids = self.ytvos.getCatIds() 
        self.vid_ids = self.ytvos.getVidIds() 
        self.vid_infos = []
        self.exp_infos = load_expressions(exp_file)
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            filename = vid_info['file_names'][0].split('/')[0]
            exps = self.exp_infos[filename]['expressions']
            for frame_id in range(len(vid_info['filenames'])):
                for exp_id in range(len(exps)):
                    self.img_ids.append((idx, frame_id, exp_id, 0))

        # a2d
        self.paths = paths
        self.col_path = os.path.join(paths['annotation_path'], 'col')  # rgb
        self._read_video_info()
        self._read_dataset_samples()
        self.audio_a2d = 'data/a2d_j_audio_feature'
        self.videos = self.train_videos
        self.samples = self.train_samples
        for sample in self.samples:
            self.img_ids.append(sample)


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if self.img_ids[idx][-1] == 0:
            vid, frame_id, exp_id, flag = self.img_ids[idx]
            vid_id = self.vid_infos[vid]['id']
            img = []
            vid_len = len(self.vid_infos[vid]['file_names'])
            inds = list(range(self.num_frames))
            inds = [i%vid_len for i in inds][::-1]
            # if random 
            # random.shuffle(inds)

            filename = self.vid_infos[vid]['file_names'][0].split('/')[0]
            a_filename = filename + '_' + str(exp_id) + '.npy'
            audio = np.load(os.path.join(self.audio_ytvos, a_filename))
            audio = audio.transpose()
            audio = torch.as_tensor(audio, dtype=torch.float32)

            exps = self.exp_infos[filename]['expressions']
            obj_id = int(exps[exp_id]['obj_id'])

            for j in range(self.num_frames):
                img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
                img.append(Image.open(img_path).convert('RGB'))

            ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
            # if obj_id > len(ann_ids):
            #     print('--------------------------', filename, obj_id)
            ann_ids = [ann_ids[obj_id-1]]

            target = self.ytvos.loadAnns(ann_ids)
            target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
            target = self.prepare(img[0], target, inds, self.num_frames)
            if self._transforms is not None:
                img, target = self._transforms(img, target)

            return torch.cat(img,dim=0), audio, target
        
        else:
            index = idx
            video_id, instance_id, frame_idx, flag = self.img_ids[index]
            frame_idx = int(frame_idx)
            h5_path = os.path.join('data/a2d/a2d_annotation_with_instances', video_id,
                                '%05d.h5' % (frame_idx + 1))
            if not os.path.exists(h5_path):
                h5_path = os.path.join('data/a2d/a2d_annotation_with_instances', video_id,
                                    '%05d.h5' % (24 + 1))
            frame_path = os.path.join('data/a2d/Release/pngs320H', video_id)

            frames = list(map(lambda x: os.path.join(frame_path, x),
                            sorted(os.listdir(frame_path))))
            # print(len(frames), self.videos['num_frames'])

            assert len(frames) == self.videos[video_id]['num_frames']

            all_frames = []
            mid_frame = (self.num_frames-1)//2
            for i in range(self.num_frames):
                all_frames.append(frame_idx-mid_frame+i)
            for i in range(len(all_frames)):
                if all_frames[i] < 0:
                    all_frames[i] = 0
                elif all_frames[i] >= len(frames):
                    all_frames[i] = len(frames) - 1
            all_frames = np.asarray(frames)[all_frames]

            img = []
            for i in all_frames:
                img.append(Image.open(i).convert('RGB'))

            # audio feature
            a_filename = video_id+'_'+instance_id+'.npy'
            audio = np.load(os.path.join(self.audio_a2d, a_filename))
            audio = audio.transpose()
            audio = torch.as_tensor(audio, dtype=torch.float32)

            # fine-grained mask
            with h5py.File(h5_path, mode='r') as fp:
                instance = np.asarray(fp['instance'])
                all_masks = np.asarray(fp['reMask'])
                if len(all_masks.shape) == 3 and instance.shape[0] != all_masks.shape[0]:
                    print(video_id, frame_idx + 1, instance.shape, all_masks.shape)

                all_boxes = np.asarray(fp['reBBox']).transpose([1, 0])  # [w_min, h_min, w_max, h_max]
                all_ids = np.asarray(fp['id'])
                # if video_id == 'EadxBPmQvtg' and frame_idx == 24:
                #     instance = instance[:-1]
                assert len(all_masks.shape) == 2 or len(all_masks.shape) == 3
                if len(all_masks.shape) == 2:
                    mask = all_masks[np.newaxis]
                    class_id = int(all_ids[0][0])
                    coarse_gt_box = all_boxes[0]
                else:
                    instance_id = int(instance_id)
                    idx = np.where(instance == instance_id)[0][0]

                    mask = all_masks[idx]
                    coarse_gt_box = all_boxes[idx]
                    class_id = int(all_ids[0][idx])
                    mask = mask[np.newaxis]

                assert len(mask.shape) == 3
                assert mask.shape[0] > 0

                fine_gt_mask = np.transpose(np.asarray(mask), (0, 2, 1))[0]

            area = np.sum(fine_gt_mask)
            w, h = img[0].size
            target = {}
            target['boxes'] = torch.from_numpy(coarse_gt_box).float().unsqueeze(0)
            target['labels'] = torch.from_numpy(np.array([class_id]))
            target['masks'] = torch.from_numpy(fine_gt_mask).unsqueeze(0)
            target['image_id'] = torch.tensor([index])
            target['valid'] = torch.tensor([1])
            target['area'] = torch.tensor([int(area)])
            target['iscrowd'] = torch.tensor([0])
            target["orig_size"] = torch.as_tensor([int(h), int(w)])
            target["size"] = torch.as_tensor([int(h), int(w)])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            return torch.cat(img,dim=0), audio, target
    
    def _read_video_info(self):
        self.train_videos, self.test_videos = {}, {}
        with open(self.paths['videoset_path'], newline='') as fp:
            reader = csv.reader(fp, delimiter=',')
            for row in reader:
                frame_idx = list(map(lambda x: int(x[:-4]) - 1,
                                     os.listdir(os.path.join(self.col_path, row[0]))))
                frame_idx = sorted(frame_idx)
                # print(frame_idx)
                # exit(0)
                video_info = {
                    'label': int(row[1]),
                    'timestamps': [row[2], row[3]],
                    'size': [int(row[4]), int(row[5])],  # [height, width]
                    'num_frames': int(row[6]),
                    'num_annotations': int(row[7]),
                    'frame_idx': frame_idx,
                }
                if int(row[8]) == 0:
                    self.train_videos[row[0]] = video_info
                else:
                    self.test_videos[row[0]] = video_info

    def _read_dataset_samples(self):
        self.train_samples, self.test_samples = [], []
        self.train_videos_set = set()
        self.test_videos_set = set()
        self.all_query = set()
        with open(self.paths['sample_path'], newline='') as fp:
            reader = csv.DictReader(fp)
            rows = []
            for row in reader:
                rows.append(row)
            for row in rows:
                if row['video_id'] in self.train_videos:
                    self.train_samples.append([row['video_id'], row['instance_id'], row['frame_idx'], 1])
                    self.train_videos_set.add(row['video_id'])
                else:
                    # if l[len(l) >> 1] != row['frame_idx']:
                    #     continue
                    self.test_samples.append([row['video_id'], row['instance_id'], row['frame_idx'], 1])
                    self.test_videos_set.add(row['video_id'])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 36, row['query']])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 61, row['query']])
                self.all_query.add((row['video_id'], row['query']))
        print('number of sentences: {}'.format(len(self.all_query)))
        print('videos for training: {}, videos for testing: {}'.format(len(self.train_videos_set),
                                                                       len(self.test_videos_set)))
        print(
            'samples for training: {}, samples for testing: {}'.format(len(self.train_samples), len(self.test_samples)))
        # exit(0)


def load_expressions(exp_file):
    with open(exp_file) as f:
        videos = json.load(f)['videos']
    exp_infos = {}
    for k, v in videos.items():
        exps = v['expressions']
        exp_list = []
        for exp in exps.values():
            exp_list.append(exp)
        exp_infos[k] = {"expressions": exp_list, "frames": v['frames']}
    return exp_infos

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, inds, num_frames):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            for j in range(num_frames):
                bbox = ann['bboxes'][frame_id-inds[j]]
                areas = ann['areas'][frame_id-inds[j]]
                segm = ann['segmentations'][frame_id-inds[j]]
                clas = ann["category_id"]
                # for empty boxes
                if bbox is None:
                    bbox = [0,0,0,0]
                    areas = 0
                    valid.append(0)
                    clas = 0
                else:
                    valid.append(1)
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes.append(bbox)
                area.append(areas)
                segmentations.append(segm)
                classes.append(clas)
                iscrowd.append(crowd)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area) 
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return  target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
                     # To suit the GPU memory the scale might be different
                     T.RandomResize([300], max_size=540),#for r50
                     #T.RandomResize([280], max_size=504),#for r101
            ]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    mode = 'instances'

    PATHS = {
        "train": (root / "train/JPEGImages", root / "train/Annotations", root /  f'ann/{mode}_train_sub.json', root / "meta_expressions/train/meta_expressions.json"),
        "val": (root / "valid/JPEGImages", root /  f'ann/{mode}_valid_sub.json'),
    }
    img_folder, mask_folder, ann_file, exp_file = PATHS[image_set]

    paths = {
        "videoset_path": "data/a2d/Release/videoset.csv",
        "annotation_path": "data/a2d/Release/Annotations",
        "sample_path": "data/a2d/a2d_annotation_info.txt",
    }

    dataset = AVOSDataset(paths, img_folder, mask_folder, ann_file, exp_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, num_frames = args.num_frames)
    return dataset
