import os
from pathlib import Path
from typing import TypedDict
import scipy.io as scio
import glob
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from .modality import *


def decode_config(config):
    all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                    'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                    'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
    all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                   'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    train_form = {}
    val_form = {}
    # Limitation to actions (protocol)
    if config['protocol'] == 'protocol1':  # Daily actions
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':  # Rehabilitation actions:
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions
    # Limitation to subjects and actions (split choices)
    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(all_subjects))
            idx_train = idx[:int(np.floor(ratio*len(all_subjects)))]
            idx_val = idx[int(np.floor(ratio*len(all_subjects))):]
            subjects_train = np.array(all_subjects)[idx_train].tolist()
            subjects_val = np.array(all_subjects)[idx_val].tolist()
            for subject in all_subjects:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1
    elif config['split_to_use'] == 'cross_scene_split':
        subjects_train = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
                          'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20',
                          'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']
        subjects_val = ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    elif config['split_to_use'] == 'cross_subject_split':
        subjects_train = config['cross_subject_split']['train_dataset']['subjects']
        subjects_val = config['cross_subject_split']['val_dataset']['subjects']
        for subject in subjects_train:
            train_form[subject] = actions
        for subject in subjects_val:
            val_form[subject] = actions
    else:
        subjects_train = config['manual_split']['train_dataset']['subjects']
        subjects_val = config['manual_split']['val_dataset']['subjects']
        actions_train = config['manual_split']['train_dataset']['actions']
        actions_val = config['manual_split']['val_dataset']['actions']
        for subject in subjects_train:
            train_form[subject] = actions_train
        for subject in subjects_val:
            val_form[subject] = actions_val

    dataset_config = {'train_dataset': {'modalities': config['modalities'],
                                        'split': 'training',
                                        'data_form': train_form
                                        },
                      'val_dataset': {'modalities': config['modalities'],
                                      'split': 'validation',
                                      'data_form': val_form}}
    return dataset_config


class MMFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities:list[Modality] = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith("."):
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
    def __init__(self, data_base, modalities:list[list], split, data_form):
        self.data_base = data_base
        self.modalities = [MODALITY_MAP[m_str] for m_str in modalities]
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()

    def get_scene(self, subject):
        scenes = {range(1, 11): 'E01', range(11, 21): 'E02', range(21, 31): 'E03', range(31, 41): 'E04'}
        for key, value in scenes.items():
            if int(subject[1:]) in key:
                return value
        raise ValueError('Subject does not exist in this dataset.')

    def load_data(self):
        raise NotImplementedError("use MMFI_DatasetFrame or MMFI_DatasetSequence")

    def __getitem__(self, idx):
        raise NotImplementedError("use MMFI_DatasetFrame or MMFI_DatasetSequence")

    def __len__(self):
        return len(self.data_list)

class MMFI_DatasetFrame(MMFi_Dataset):
    def load_data(self):
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                frame_num = 297
                for idx in range(frame_num):
                    data_dict = {'modalities': [m.name for m in self.modalities],
                                    'scene': self.get_scene(subject),
                                    'subject': subject,
                                    'action': action,
                                    'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                            action, 'ground_truth.npy'),
                                    'idx': idx
                                    }
                    data_valid = True
                    for mod in self.modalities:
                        data_dict[mod.name+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, mod.name, "frame{:03d}".format(idx+1) + mod.file_ending)
                    if data_valid:
                        data_info.append(data_dict)
        return data_info

    def __getitem__(self, idx):
        item = self.data_list[idx]

        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)
        sample = {'modalities': item['modalities'],
                    'scene': item['scene'],
                    'subject': item['subject'],
                    'action': item['action'],
                    'idx': item['idx'],
                    'output': gt_torch[item['idx']]
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
        for subject, actions in self.data_source.items():
            for action in actions:
                data_dict = {'modalities': [m.name for m in self.modalities],
                                'scene': self.get_scene(subject),
                                'subject': subject,
                                'action': action,
                                'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                        action, 'ground_truth.npy')
                                }
                for mod in self.modalities:
                    data_dict[mod.name+'_path'] = os.path.join(self.data_base.data_root, self.get_scene(subject), subject,
                                                        action, mod.name)
                data_info.append(data_dict)

        return data_info

    def __getitem__(self, idx):
        item = self.data_list[idx]
        gt_numpy = np.load(item['gt_path'])
        gt_torch = torch.from_numpy(gt_numpy)
        sample = {'modalities': item['modalities'],
                    'scene': item['scene'],
                    'subject': item['subject'],
                    'action': item['action'],
                    'output': gt_torch
                    }
        for mod in self.modalities:
            data_path = item[mod.name+'_path']
            data_mod = mod.read_dir(data_path)
            sample[mod.name+'_path'] = item[mod.name+'_path']
            sample[mod.name] = data_mod

        return sample

def make_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)
    config_dataset = decode_config(config)
    if config["data_unit"]== "frame":
        train_dataset = MMFI_DatasetFrame(database, **config_dataset['train_dataset'])
        val_dataset = MMFI_DatasetFrame(database, **config_dataset['val_dataset'])
    elif config["data_unit"]== "sequence":
        train_dataset = MMFI_DatasetSequence(database, **config_dataset['train_dataset'])
        val_dataset = MMFI_DatasetSequence(database, **config_dataset['val_dataset'])
    else:
        raise ValueError("invalid data_unit")

    return train_dataset, val_dataset


def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''

    batch_data = {'modalities': batch[0]['modalities'],
                  'scene': [sample['scene'] for sample in batch],
                  'subject': [sample['subject'] for sample in batch],
                  'action': [sample['action'] for sample in batch],
                  'idx': [sample['idx'] for sample in batch] if 'idx' in batch[0] else None
                  }
    _output = [np.array(sample['output']) for sample in batch]
    _output = torch.FloatTensor(np.array(_output))
    batch_data['output'] = _output

    for mod in batch_data['modalities']:
        mod = MODALITY_MAP[mod]
        if mod.name in ['mmwave', 'lidar']:
            _input = [torch.Tensor(sample[mod.name]) for sample in batch]
            _input = torch.nn.utils.rnn.pad_sequence(_input)
            _input = _input.permute(1, 0, 2)
            batch_data[mod] = _input
        else:
            _input = [np.array(sample[mod.name]) for sample in batch]
            _input = torch.FloatTensor(np.array(_input))
            batch_data[mod.name] = _input

    return batch_data

def make_dataloader(dataset, is_training, generator, batch_size, collate_fn_padd = collate_fn_padd):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padd,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator
    )
    return loader


