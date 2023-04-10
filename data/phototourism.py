import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle
from . import base
import camera
from util import log,debug

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        super().__init__(opt,split)
        self.opt = opt
        self.root_dir = os.path.join("/hub_data/injae/nerf/phototourism", opt.data.scene) # /hub_data/injae/nerf/phototourism
        self.temp_dir = os.path.join("/hub_data/minhyuk/cache", opt.data.scene) # /hub_data/minhyuk/cache
        self.path_image = os.path.join(self.root_dir, 'dense/images')
        self.split=split
        self.read_meta(opt)

    def read_meta(self, opt):
        
        with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
            all_img_ids = pickle.load(f)
        if self.split == 'train':
            with open(os.path.join(self.temp_dir, 'img_ids_train.pkl'), 'rb') as f:
                img_ids = pickle.load(f)
        else:
            with open(os.path.join(self.temp_dir, 'img_ids_test.pkl'), 'rb') as f:
                img_ids = pickle.load(f)           
        with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
            image_paths_dict = pickle.load(f)
        with open(os.path.join(self.root_dir, f'cache/Ks{opt.img_downscale}.pkl'), 'rb') as f:
            Ks_dict = pickle.load(f)
        poses = torch.tensor(np.load(os.path.join(self.root_dir, 'cache/poses.npy'))).float()
        image_paths, Ks, poses_dict = [], [], {}
        for i, id_ in enumerate(all_img_ids):
            poses_dict[id_] = poses[i]
        poses = []
        for id_ in img_ids:
            image_paths.append(image_paths_dict[id_])
            Ks.append(Ks_dict[id_])
            poses.append(poses_dict[id_])
        self.list = list(zip(image_paths, poses, Ks))
        # manually split train/val subsets
        # self.num_val_split = int(len(self.list)*opt.data.val_ratio)
        # self.list = self.list[:-self.num_val_split] if self.split=="train" else self.list[-self.num_val_split:]
        
    def __len__(self):
        return len(self.list)

    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        image, w, h = self.get_image(opt,idx)
        intr,pose = self.get_camera(opt,idx)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
            W = w,
            H = h
        )
        return sample
    
    def get_size(self, opt, idx):
        w, h = self.list[idx][3], self.list[idx][4]
        return w, h

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image, self.list[idx][0])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        w, h = image.width, image.height 
        image = image.resize((w // opt.img_downscale, h // opt.img_downscale))
        image_tensor = torchvision_F.to_tensor(image)
        w, h = w // opt.img_downscale, h // opt.img_downscale
        return image_tensor, w, h

    def get_camera(self,opt,idx):
        pose_raw = self.list[idx][1]
        intr = torch.tensor(self.list[idx][2]).float()
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        pose = camera.pose.compose([pose_flip,pose])
        return pose

    def get_all_camera_poses(self,opt):
        pose_raw_all = [tup[1] for tup in self.list]
        pose_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_all