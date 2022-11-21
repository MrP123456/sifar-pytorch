import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import multiprocessing
import sys

from video_transforms import (GroupRandomHorizontalFlip, GroupOverSample,
                              GroupMultiScaleCrop, GroupScale, GroupCenterCrop, GroupRandomCrop,
                              GroupNormalize, Stack, ToTorchFormatTensor, GroupRandomScale)


def build_dataflow(dataset, is_train, batch_size, workers=36, is_distributed=False):
    workers = min(workers, multiprocessing.cpu_count())
    shuffle = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    if is_train:
        shuffle = sampler is None

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                              num_workers=workers, pin_memory=True, sampler=sampler)

    return data_loader


class VideoRecord(object):
    def __init__(self, elements):
        self.path = elements[0]
        self.video_id = os.path.basename(self.path)
        self.start_frame = int(elements[1])
        self.end_frame = int(elements[2])
        self.label = int(elements[3])

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1


def get_augment(img_size=224, mean=None, std=None, version='v1'):
    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std
    augments = []
    assert version in ['v1', 'v2']
    if version == 'v1':
        augments += [
            GroupMultiScaleCrop(img_size, [1, .875, .75, .66])
        ]
    else:
        augments += [
            GroupRandomScale([256, 320]),
            GroupRandomCrop(img_size),
        ]
    augments += [
        Stack(threed_data=True),
        ToTorchFormatTensor(num_clips_crops=1),
        GroupNormalize(mean=mean, std=std, threed_data=True)
    ]
    return transforms.Compose(augments)


class VideoDataSet(Dataset):
    def __init__(self, root_path, num_groups=64, frames_per_group=1, sample_offset=0, num_clips=1, modality='rgb',
                 dense_sampling=False, fixed_offset=True, image_tmpl='{:05d}.jpg', transform=None, is_train=True,
                 test_mode=False, seperator=' ', filter_video=0, num_classes=None, whole_video=False, fps=29.97,
                 audio_length=1.28, resampling_rate=24000):
        """
        Args:
            root_path (str): the file path to the root of video folder
            num_groups (int): number of frames per data sample
            frames_per_group (int): number of frames within one group
            sample_offset (int): used in validation/test, the offset when sampling frames from a group
            modality (str): rgb or flow
            dense_sampling (bool): dense sampling in I3D
            fixed_offset (bool): used for generating the same videos used in TSM
            image_tmpl (str): template of image ids
            transform: the transformer for preprocessing
            is_train (bool): shuffle the video but keep the causality
            test_mode (bool): testing mode, no label
            whole_video (bool): take whole video
            fps (float): frame rate per second, used to localize sound when frame idx is selected.
            audio_length (float): the time window to extract audio feature.
            resampling_rate (int): used to resampling audio extracted from wav
        """
        self.root_path = root_path
        typ = 'train.txt' if is_train else 'val.txt'
        self.list_file = os.path.join(root_path, typ)
        self.num_groups = num_groups
        self.num_frames = num_groups
        self.frames_per_group = frames_per_group
        self.sample_freq = frames_per_group
        self.num_clips = num_clips
        self.sample_offset = sample_offset
        self.fixed_offset = fixed_offset
        self.dense_sampling = dense_sampling
        self.modality = modality.lower()
        self.image_tmpl = image_tmpl
        self.is_train = is_train
        self.test_mode = test_mode
        self.separator = seperator
        self.filter_video = filter_video
        self.num_classes = num_classes
        self.whole_video = whole_video
        self.fps = fps
        self.audio_length = audio_length
        self.resampling_rate = resampling_rate
        self.video_length = (self.num_frames * self.sample_freq) / self.fps
        self.video_list = self._parse_list()
        self.transform = transform if transform is not None else get_augment()

    def _parse_list(self):
        # folder_path, start_frame, end_frame, label_id
        video_list = []
        for x in open(self.list_file):
            elements = x.strip().split(self.separator)
            video_list.append(VideoRecord(elements))
        return video_list

    def __getitem__(self, item):
        record = self.video_list[item]
        indices = self._sample_indices(record) if self.is_train else self._get_val_indices(record)
        images = self.get_data(record, indices)
        images = self.transform(images)
        label = self.get_label(record)
        return images, label

    def _sample_indices(self, record):
        if self.dense_sampling:
            frame_idx = np.asarray(self.random_clip(record.num_frames))
        else:
            if record.num_frames < self.num_groups * self.frames_per_group:
                if self.num_groups < record.num_frames:
                    frame_idx = np.random.choice(record.num_frames, self.num_groups, replace=False)
                else:
                    frame_idx = np.random.choice(record.num_frames, self.num_groups)
            else:
                frame_idx = np.arange(0, self.num_groups) * self.frames_per_group
                offset = np.random.choice(record.num_frames // self.num_groups, self.num_groups)
                frame_idx = frame_idx + offset
            frame_idx = np.sort(frame_idx)
        frame_idx += 1
        return frame_idx

    def random_clip(self, num_frames):
        if num_frames - self.num_groups * self.frames_per_group <= 0:
            start_frame = 0
        else:
            if self.fixed_offset:
                start_frame = (num_frames - self.num_groups * self.frames_per_group) // 2
            else:
                start_frame = int(np.random.randint(0, num_frames - self.num_groups * self.frames_per_group, 1))
        frame_idx = [int(start_frame + i * self.frames_per_group) % num_frames for i in range(self.num_groups)]
        return frame_idx

    def _get_val_indices(self, record):
        if self.whole_video:
            return np.arrange(0, record.num_frames, step=self.frames_per_group) + 1
        if self.dense_sampling:
            sample_pos = max(1, 1 + record.num_frames - self.frames_per_group * self.num_groups)
            frame_idx = [(idx * sample_pos) % record.num_frames for idx in range(self.num_groups)]
        else:
            if record.num_frames < self.num_groups * self.frames_per_group:
                if self.num_groups < record.num_frames:
                    frame_idx = np.random.choice(record.num_frames, self.num_groups, replace=False)
                else:
                    frame_idx = np.random.choice(record.num_frames, self.num_groups)
            else:
                frame_idx = np.arange(0, self.num_groups) * self.frames_per_group
                offset = np.random.choice(record.num_frames // self.num_groups, self.num_groups)
                frame_idx = frame_idx + offset
            frame_idx = np.sort(frame_idx)
            frame_idx += 1
        return frame_idx

    def get_data(self, record, indices):
        images = []
        for indice in indices:
            idx = min(indice + record.start_frame + 1, record.num_frames)
            path = os.path.join(self.root_path, record.path, self.image_tmpl.format(idx))
            tmp_img = Image.open(path)
            img = tmp_img.copy()
            tmp_img.close()
            images.append(img)
        return images

    def get_label(self, record):
        return record.video_id

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    dataset_train = VideoDataSet('F:/dataset_dir', is_train=True)
    dataloader_train = build_dataflow(dataset_train, is_train=True, batch_size=8, workers=0, is_distributed=False)
    for x, y in dataloader_train:
        print(x.shape)
        sys.exit(0)
