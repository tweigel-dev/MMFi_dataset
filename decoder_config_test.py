

import pytest

from mmfi_dataset.decode_config import MMFIConfig, DatasetFragment


def test_no_intersection():
    config =  MMFIConfig(
                modalities=["wifi-csi","flow"],
                train=DatasetFragment(environments=["E01"]),
                validation=DatasetFragment(environments=["E02"]))

def test_env_intersection():
    with pytest.raises(ValueError):
        MMFIConfig(
            modalities=["wifi-csi","flow"],
            train=DatasetFragment(environments=["E01"]),
            validation=DatasetFragment(environments=["E01"]))

def test_subject_intersection():
    with pytest.raises(ValueError):
        MMFIConfig(
            modalities=["wifi-csi","flow"],
            train=DatasetFragment(environments=["E01"],subjects=["S01"]),
            validation=DatasetFragment(environments=["E01"],subjects=["S01","S04"]))

def test_subject_no_intersection():

    MMFIConfig(
            modalities=["wifi-csi","flow"],
            train=DatasetFragment(environments=["E01"],subjects=["S01"]),
            validation=DatasetFragment(environments=["E01"],subjects=["S04"]))
        

        
def test_action_intersection():
    with pytest.raises(ValueError):
        MMFIConfig(
            modalities=["wifi-csi","flow"],
            train=DatasetFragment(environments=["E01"],subjects=["S01"], actions=["A01","A02"]),
            validation=DatasetFragment(environments=["E01"],subjects=["S01"], actions=["A02"]))

def test_action_no_intersection():
    MMFIConfig(
            modalities=["wifi-csi","flow"],
            train=DatasetFragment(environments=["EO1"],subjects=["S01"], actions=["A01"]),
            validation=DatasetFragment(environments=["E01"],subjects=["S01"], actions=["A02"]))


def test_invalid_modality():
    with pytest.raises(ValueError):
        MMFIConfig(
            modalities=["wifi-csi","flow","dummy modalitiy"],
            train=DatasetFragment(environments=["EO1"],subjects=["S01"], actions=["A01"]),
            validation=DatasetFragment(environments=["E01"],subjects=["S01"], actions=["A02"]))
