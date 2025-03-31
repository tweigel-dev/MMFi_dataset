

import os
from pathlib import Path
import scipy.io as scio
import glob
import cv2
import numpy as np
from torchvision.io import read_image
import torch
from torch import Tensor



class Modality:
    def __init__(self, name: str) -> None:
        self.name = name
    @property
    def file_ending()-> str:
        raise NotImplementedError()
    def read_dir(self, path: str | Path):
        if os.path.isdir(path):
            return np.array([self.read_frame(f) for f in sorted(glob.glob(os.path.join(path, f"frame*{self.file_ending}")))])
        raise ValueError(f"Path {path} is not a dir")
    def read_frame(self, frame_path: str | Path):
        raise NotImplementedError()
    def __str__(self) -> str:
        return self.name
class KeypointModality(Modality):
    file_ending=".npy"

    def read_frame(self, frame_path: str | Path):
        return np.load(frame_path)

class DepthModality(Modality):
    file_ending=".png"

    def read_frame(self, frame_path: str | Path):
        return cv2.imread(frame_path, cv2.IMREAD_UNCHANGED) * 0.001


class ImageModality(Modality):
    file_ending=".png"
    def read_dir(self, path: str | Path):
        path = Path(str(path).replace(self.name,"rgb"))
        return super().read_dir(path)

    def read_frame(self, frame_path: str | Path):
        frame_path = Path(str(frame_path).replace(self.name,"rgb"))
        return read_image(frame_path)

class LidarModality(Modality):
    file_ending=".bin"

    def read_frame(self, frame_path: str | Path):
        with open(frame_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float64)
        return data.reshape(-1, 3)



class MmwaveModality(Modality):
    file_ending=".bin"
    def read_frame(self, frame_path: str | Path):
        with open(frame_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float64)
        return data.copy().reshape(-1, 5)


class FlowModality(Modality):
    file_ending=".pt"
    def read_frame(self, frame_path: str | Path):
        return torch.load(frame_path).to(torch.float16)


class WifiCSIModality(Modality):
    file_ending=".mat"
    def read_frame(self, frame_path: str | Path):
        data_amp = scio.loadmat(frame_path)['CSIamp']
        data_pha = scio.loadmat(frame_path)['CSIphase']
        data = torch.complex(Tensor(data_amp),Tensor(data_pha))
        return self._interpolate_nan_inf(data)


    def _interpolate_nan_inf(self, csi_frame_path: np.ndarray):
        for i in range(10):
            temp_col = csi_frame_path[:, :, i]
            if temp_col.isnan().any() or temp_col.isinf().any():
                findings = temp_col.isnan() | temp_col.isinf() 
                temp_col[findings] = temp_col[findings.logical_not()].mean()
        return csi_frame_path

class WifiCSIAmplitudeModality(WifiCSIModality):
    def read_frame(self, frame_path: str | Path):
        data_amp = scio.loadmat(frame_path)['CSIamp']
        return self._interpolate_nan_inf(Tensor(data_amp))

class WifiCSIPhaseModality(WifiCSIModality):
    def read_frame(self, frame_path: str | Path):
        data_pha = scio.loadmat(frame_path)['CSIphase']
        return self._interpolate_nan_inf(Tensor(data_pha))

MODALITY_MAP:dict[str,Modality] = {
    'infra1': KeypointModality('infra1'),
    'infra2': KeypointModality('infra2'),
    'rgb': KeypointModality('rgb'),
    'depth': DepthModality('depth'),
    'image': ImageModality('image'),
    'lidar': LidarModality('lidar'),
    'mmwave': MmwaveModality('mmwave'),
    'wifi-csi': WifiCSIModality('wifi-csi'),
    'wifi-csi-amp': WifiCSIAmplitudeModality('wifi-csi-amp'),
    'wifi-csi-pha': WifiCSIPhaseModality('wifi-csi-pha'),
    'flow': FlowModality('flow')
}