import csv
import os
import random

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import slic
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torchvision.transforms.functional as F

import datasets.transforms as T


class A2DDataset(Dataset):
    # def __init__(self, videos, samples, word2vec, bert, id2idx, args, transforms=None, train=False):
    def __init__(self, image_set, args, num_frames):
        self.args = args
        self.col_path = os.path.join(args['annotation_path'], 'col')  # rgb
        self._read_video_info()
        self._read_dataset_samples()
        self.audio_file = 'data/a2d_j_audio_feature'
        self.num_frames = num_frames
        self.videos = self.train_videos
        self.samples = self.train_samples
        self._transforms = make_coco_transforms(image_set)
        # np.random.seed(88)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_id, instance_id, frame_idx = self.samples[index]
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
        audio = np.load(os.path.join(self.audio_file, a_filename))
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

            fine_gt_mask = np.transpose(np.asarray(mask), (0, 2, 1))[0]  # 320*568

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
        with open(self.args['videoset_path'], newline='') as fp:
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
        with open(self.args['sample_path'], newline='') as fp:
            reader = csv.DictReader(fp)
            rows = []
            for row in reader:
                rows.append(row)
            for row in rows:
                if row['video_id'] in self.train_videos:
                    self.train_samples.append([row['video_id'], row['instance_id'], row['frame_idx']])
                    self.train_videos_set.add(row['video_id'])
                else:
                    # if l[len(l) >> 1] != row['frame_idx']:
                    #     continue
                    self.test_samples.append([row['video_id'], row['instance_id'], row['frame_idx']])
                    self.test_videos_set.add(row['video_id'])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 36, row['query']])
                    # self.test_samples.append([row['video_id'], row['instance_id'], 61, row['query']])
                self.all_query.add((row['video_id'], row['query']))
        print('number of sentences: {}'.format(len(self.all_query)))
        print('videos for training: {}, videos for testing: {}'.format(len(self.train_videos_set),
                                                                       len(self.test_videos_set)))
        print(
            'samples for training: {}, samples for testing: {}'.format(len(self.train_samples), len(self.test_samples)))


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
    paths = {
        "videoset_path": "data/a2d/Release/videoset.csv",
        "annotation_path": "data/a2d/Release/Annotations",
        "sample_path": "data/a2d/a2d_annotation_info.txt",
    }
    dataset = A2DDataset(image_set, paths, num_frames = args.num_frames)
    return dataset

