import os
import sys
import random
import torch
import numpy as np
import argparse
BASEPATH = os.path.dirname(__file__)
from os.path import join as pjoin
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from torch.utils.data import Dataset, DataLoader
from utils.animation_data import AnimationData
from utils.load_skeleton import Skel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_labels = ["walk", "run", "jump", "punch", "kick"]

style_labels = ["angry", "childlike", "depressed", "old", "proud", "sexy", "strutting"]

class MotionDataset(Dataset):
    def __init__(self, args, subset_name="train"):
        super(MotionDataset, self).__init__()

        self.args = args
        self.skel = Skel()

        dataset = np.load(args.data_path, allow_pickle=True)[subset_name].item()
        motions, labels, metas = dataset["motion"], dataset["style"], dataset["meta"]

        self.label_i = labels
        self.len = 0
        self.metas = [{key: metas[key][i] for key in metas.keys()} for i in range(self.len)]
        self.motion_i, self.foot_i = [], []
        self.joint_rotations, self.joint_positions, self.joint_velocities, self.styles, self.contents, self.roots, self.contacts = [], [], [], [], [], [], []

        self.labels = []
        self.data_dict = {}

        for i, motion in enumerate(motions):
            episode_length = motion.shape[0]
            content = metas["content"][i]
            style = metas["style"][i]
            if content != "trans":
                self.len += 1
                
                anim = AnimationData(motion, skel=self.skel)
                self.motion_i.append(anim)
                self.joint_rotations.append(anim.get_joint_rotation())
                self.joint_positions.append(anim.get_joint_position())
                self.joint_velocities.append(anim.get_joint_velocity())
                style_array = np.zeros(len(style_labels))
                if style != "neutral":
                    style_array[style_labels.index(style)] = 1.0
                self.styles.append(np.tile([style_array],(episode_length,1)))
                content_index = content_labels.index(content)
                self.contents.append(np.tile([np.eye(len(content_labels))[content_index]],(episode_length, 1)))
                self.contacts.append(anim.get_foot_contact(transpose=False))  # [T, 4]
                self.roots.append(anim.get_root_posrot())

        self.dim_dict = {
            "rotation": self.joint_rotations[0].shape[-1],
            "position": self.joint_positions[0].shape[-1],
            "velocity": self.joint_velocities[0].shape[-1],
            "style": self.styles[0].shape[-1], 
            "content": self.contents[0].shape[-1],
            "contact": self.contacts[0].shape[-1],
            "root": self.roots[0].shape[-1]
        }
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input_style = self.styles[index][0,:]
        episode_length = self.styles[index].shape[0]
        output_style = np.zeros(len(style_labels))
        if input_style.sum() == 0.0:
            output_style[np.random.randint(0, len(style_labels))] = 1
        output_style = np.tile([output_style],(episode_length,1))
        data = {
            "rotation": torch.FloatTensor(self.joint_rotations[index]).to(device),
            "position": torch.FloatTensor(self.joint_positions[index]).to(device),
            "velocity": torch.FloatTensor(self.joint_velocities[index]).to(device),
            "content": torch.FloatTensor(self.contents[index]).to(device),
            "contact": torch.FloatTensor(self.contacts[index]).to(device),
            "root": torch.FloatTensor(self.roots[index]).to(device),
            "input_style": torch.FloatTensor(self.styles[index]).to(device),
            "transferred_style": torch.FloatTensor(output_style).to(device),
            "content_index": torch.LongTensor(np.argwhere(self.contents[index][0]==1)[0]).to(device)
        }

        return data