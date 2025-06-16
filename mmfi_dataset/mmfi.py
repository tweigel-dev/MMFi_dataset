import os
from pathlib import Path
from typing import TypedDict
import scipy.io as scio
import glob
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from mmfi_dataset.decode_config import DatasetFragment

from .modality import *



class MMFi_Database:
    def __init__(self, data_root):
        self.data_root:Path = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities:list[Modality] = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith((".","_")) :
                continue
            if os.path.isfile( os.path.join(self.data_root, scene)):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                if subject.startswith("."):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    if action.startswith("."):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in self.modalities:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality.name)
                        self.scenes[scene][subject][action][modality.name] = data_path
                        self.subjects[subject][action][modality.name] = data_path
                        self.actions[action][scene][subject][modality.name] = data_path

class MMFi_Dataset(Dataset):
    def __init__(self, database:MMFi_Database, modalities:list[str], fragment:DatasetFragment):
        self.database = database
        self.modalities = [MODALITY_MAP[m_str] for m_str in modalities]
        self.fragment:DatasetFragment = fragment
        self.data_list = self.load_data()


    def load_data(self):
        raise NotImplementedError("use MMFI_DatasetFrame or MMFI_DatasetSequence")

    def __getitem__(self, idx):
        raise NotImplementedError("use MMFI_DatasetFrame or MMFI_DatasetSequence")

    def __len__(self):
        return len(self.data_list)

class MMFI_DatasetFrame(MMFi_Dataset):
    def load_data(self):
        data_info = []
        for relative_path in self.fragment.create_tree():
            if not (self.database.data_root/relative_path).exists():
                continue
            frame_num = len(list((self.database.data_root/relative_path/self.modalities[0].name).glob("*")))
            for idx in range(frame_num):
                data_dict = {'modalities': [m.name for m in self.modalities],
                                'idx': idx
                                }
                data_valid = True
                for mod in self.modalities:
                    data_dict[mod.name+'_path'] = self.database.data_root/relative_path/ mod.name/f"frame{idx+1:03d}{mod.file_ending}"
                    if not mod.exists(data_dict[mod.name+'_path']):
                        data_valid = False
                if data_valid:
                    data_info.append(data_dict)
        return data_info

    def __getitem__(self, idx):
        item = self.data_list[idx]

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)
        sample = {'modalities': item['modalities'],
                    'idx': item['idx'],
                    }
        for mod in self.modalities:
            data_path = item[mod.name + '_path']
            data_mod = mod.read_frame(data_path)
            sample[mod.name+'_path'] = item[mod.name+'_path']
            sample[mod.name] = data_mod

        return sample


class MMFI_DatasetSequence(MMFi_Dataset):
    def load_data(self):
        data_info = []
        for relative_path in self.fragment.create_tree():
            data_dict = {'modalities': [m.name for m in self.modalities]}
            for mod in self.modalities:
                data_dict[mod.name+'_path'] = self.database.data_root/relative_path/ mod.name
            data_info.append(data_dict)

        return data_info

    def __getitem__(self, idx):
        item = self.data_list[idx]
        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)
        sample = {'modalities': item['modalities']}
        for mod in self.modalities:
            data_path = item[mod.name+'_path']
            data_mod = mod.read_dir(data_path)
            sample[mod.name+'_path'] = item[mod.name+'_path']
            sample[mod.name] = data_mod

        return sample



def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''

    batch_data = {'modalities': batch[0]['modalities'],
                  'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
                  }

    for mod in batch_data['modalities']:
        batch_data[mod+"_path"] = [sample[mod+"_path"] for sample in batch if mod+"_path" in sample],
        mod = MODALITY_MAP[mod]
        if mod.name in ['mmwave', 'lidar']:
            _input = [torch.Tensor(sample[mod.name]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input)
            _input = _input.permute(1, 0, 2)
            batch_data[mod] = _input
        else:
            if not mod.name in  batch[0]:
                continue
            batch_data[mod.name] = torch.concat([sample[mod.name][None] for sample in batch if mod.name in sample])

    return batch_data



