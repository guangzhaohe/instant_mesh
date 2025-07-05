import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path
from typing import List, Dict

from src.utils.train_util import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class NeurokOverfitData(Dataset):
    def __init__(self,
        root_dir='data/table_seq_proc/multi_view',
        validation=False,
    ):
        self.root_dir = root_dir

        # Gather image paths for all frames
        data = []
        frame_dirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for frame_id, frame_dirs in enumerate(frame_dirs):
            data.append([])
            for view_id in range(6):
                image_path_i = os.path.join(frame_dirs, f'{view_id:03d}.png')
                data[-1].append(image_path_i)
        self.data = data

        # Construct training/inference pairs
        data_length = len(self.data)
        frame_indices = np.arange(data_length)
        pairs = np.stack((
            np.tile(frame_indices[None, :], (data_length, 1)),
            np.tile(frame_indices[:, None], (1, data_length))
        ), axis=-1).reshape(-1, 2)  # n_f * n_f, 2
        breakpoint()  # TODO: check correctness
        
        if validation:
            self.pairs = pairs[::len(pairs) // 10]  # Uniformly sample 10 pairs for validation
        else:
            self.pairs = pairs
            
        print('============= length of dataset %d =============' % len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_frame(self, frame_paths: List[str]):
        images, alphas = [], []
        assert len(frame_paths) == 6
        bkg_color = [1., 1., 1.]
        for f_path in frame_paths:
            img, alpha = self.load_im(os.path.join(self.root_dir, f_path), bkg_color)
            images.append(img)
            alphas.append(alpha)
        return images, alphas
    
    def __getitem__(self, index):
        # Gather multi-view pair
        pair = self.pairs[index]
        fid_src, fid_tar = pair[0], pair[1]
        
        image_paths_src = self.data[fid_src]  # A list of image paths
        image_paths_tar = self.data[fid_tar]
        
        images_src, alphas_src = self.load_frame(image_paths_src)
        images_tar, alphas_tar = self.load_frame(image_paths_tar)
        
        images_src_f = torch.stack(images_src, dim=0).float()  # (6, 3, H, W)
        images_tar_f = torch.stack(images_tar, dim=0).float()  # (6, 3, H, W)
        breakpoint()  # TODO: check dim correctness

        data = {
            'src_frame_id': fid_src,
            'tar_frame_id': fid_tar,
            'src_images': images_src_f,     # (6, 3, H, W)
            'tar_images': images_tar_f,     # (6, 3, H, W)
        }

        return data
