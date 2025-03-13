

import os
from pathlib import Path
import scipy.io as scio
import glob
import cv2
import numpy as np
from torchvision.io import read_image



class Modality:
    def __init__(self, name: str) -> None:
        self.name = name
    @property
    def file_ending()-> str:
        raise NotImplementedError()
    def read_dir(self, path: str | Path):
        return np.array([self.read_frame(f) for f in sorted(glob.glob(os.path.join(path, f"frame*{self.file_ending}")))])

    def read_frame(self, frame: str | Path):
        raise NotImplementedError()
    def __str__(self) -> str:
        return self.name
class KeypointModality(Modality):
    file_ending=".npy"

    def read_frame(self, frame: str | Path):
        return np.load(frame)

class DepthModality(Modality):
    file_ending=".png"

    def read_frame(self, frame: str | Path):
        return cv2.imread(frame, cv2.IMREAD_UNCHANGED) * 0.001


class ImageModality(Modality):
    file_ending=".png"

    def read_frame(self, frame: str | Path):
        return read_image(frame)

class LidarModality(Modality):
    file_ending=".bin"

    def read_frame(self, frame: str | Path):
        with open(frame, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float64)
        return data.reshape(-1, 3)



class MmwaveModality(Modality):
    file_ending=".bin"
    def read_frame(self, frame: str | Path):
        with open(frame, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float64)
        return data.copy().reshape(-1, 5)



class WifiCSIModality(Modality):
    file_ending=".mat"
    def read_frame(self, frame: str | Path):
        data_amp = scio.loadmat(frame)['CSIamp']
        data_pha = scio.loadmat(frame)['CSIphase']
        data = np.vectorize(complex)(data_amp, data_pha)
        return self._interpolate_nan_inf(data)


    def _interpolate_nan_inf(self, csi_frame: np.ndarray):
        for i in range(10):
            temp_col = csi_frame[:, :, i]
            if np.isnan(temp_col).any():
                temp_col[np.isnan(temp_col)] = np.nanmean(temp_col)
        return csi_frame

class WifiCSIAmplitudeModality(WifiCSIModality):
    def read_frame(self, frame: str | Path):
        data_amp = scio.loadmat(frame)['CSIamp']
        return self._interpolate_nan_inf(data_amp)

class WifiCSIPhaseModality(WifiCSIModality):
    def read_frame(self, frame: str | Path):
        data_pha = scio.loadmat(frame)['CSIphase']
        return self._interpolate_nan_inf(data_pha)

MODALITY_MAP:dict[str,Modality] = {
    'infra1': KeypointModality('infra1', '.npy'),
    'infra2': KeypointModality('infra2', '.npy'),
    'rgb': KeypointModality('rgb', '.npy'),
    'depth': DepthModality('depth', '.png'),
    'image': ImageModality('rgb', '.png'),
    'lidar': LidarModality('lidar', '.bin'),
    'mmwave': MmwaveModality('mmwave', '.bin'),
    'wifi-csi': WifiCSIModality('wifi-csi', '.mat'),
    'wifi-csi-amp': WifiCSIAmplitudeModality('wifi-csi-amp', '.mat'),
    'wifi-csi-pha': WifiCSIPhaseModality('wifi-csi-pha', '.mat')
}