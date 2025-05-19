

from pathlib import Path
from typing import Optional
import numpy
from pydantic import BaseModel, field_serializer, field_validator, model_validator
import yaml
from .modality import MODALITY_MAP
_all_actions = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
_all_subjects = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10', 'S11', 'S12', 'S13', 'S14',
                'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28',
                'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']
_all_environments = ["E01","E02","E03","E04"]






class DatasetFragment(BaseModel):
    environments: list[str] = _all_environments
    subjects: list[str] = _all_subjects
    actions: list[str] = _all_actions
    batch_size:int = 1
    def create_tree(self)-> set[str]:
        result = []
        for env in self.environments:
            for subject in self.subjects:
                for action in self.actions:
                    result.append(f"{env}/{subject}/{action}")
        return set(result)


class EmptyFragment(DatasetFragment):
    environments: list[str] = []
    subjects: list[str] = []
    actions: list[str] = []





class MMFIConfig(BaseModel):
    modalities : list[str]
    dataset_root: Optional[str] = None
    train : DatasetFragment
    validation: DatasetFragment
    test: Optional[DatasetFragment]  = EmptyFragment()
    seed:int = numpy.random.randint(1,9999)
    @model_validator(mode='after')
    def _validate_no_overlapp(self):
        train_tree = self.train.create_tree()
        validation_tree = self.validation.create_tree()
        test_tree = self.test.create_tree()
        if train_tree & validation_tree or train_tree & test_tree or validation_tree & test_tree:
            intersection = train_tree & validation_tree | train_tree & test_tree | validation_tree & test_tree
            raise ValueError(f'in this config is an invalid intersection {intersection}')

    @field_validator('modalities')
    @classmethod
    def _validate_modalities(cls, modalities):
        if not all(m in MODALITY_MAP for m in modalities):
            raise ValueError(f'unknown modality {set(modalities)-(set(MODALITY_MAP.keys()))}')
        return modalities

    @field_validator('dataset_root')
    @classmethod
    def _validate_modalities(cls, dataset_root:str):
        path = Path(dataset_root)
        if not path.is_dir():
            raise ValueError(f'dataset_root is not a dir {dataset_root}')
        return path
    @field_serializer("dataset_root")
    def dataset_root_ser(self, dataset_root):
        return str(dataset_root)

    @classmethod
    def load(cls, path:Path):
        with open(path,"r") as file:
            dict = yaml.safe_load(file)
            config = MMFIConfig(**dict)
        return config
    
    def save(self, path:Path):
        with open(path, "w") as file:
            yaml.dump(self.model_dump(),file,default_flow_style=None)