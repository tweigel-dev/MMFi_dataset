
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from mmfi_dataset.mmfi import  collate_fn_padd
from mmfi_dataset.mmfi import MMFi_Database, MMFI_DatasetFrame, MMFI_DatasetSequence
from mmfi_dataset.decode_config import MMFIConfig,DatasetFragment

if __name__ == '__main__':
    dataset_config = MMFIConfig(
        dataset_root="./dataset/mmfi",
        modalities=["wifi-csi","rgb"],
        train=DatasetFragment(
            environments=["EO1"],
            # subjects default all
            actions=["A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19", "A20", "A21"]
        ),
        validation=DatasetFragment(
            environments=["EO1"],
            # subjects default all
            actions= ["A22", "A23", "A24", "A25", "A26", "A27"]
        )
    )
    database = MMFi_Database(dataset_config.dataset_root)
    train_dataset = MMFI_DatasetFrame(database,dataset_config.modalities,dataset_config.train) 
    val_dataset,  = MMFI_DatasetFrame(database,dataset_config.modalities,dataset_config.train) 
    rng_generator = torch.manual_seed(dataset_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        collate_fn=collate_fn_padd,
        shuffle=True,
        drop_last=True,
        generator=rng_generator,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        collate_fn=collate_fn_padd,
        shuffle=False,
        drop_last=False,
        generator=rng_generator,
    )
    #Just an example for illustration.
    for batch_idx, batch_data in enumerate(train_loader):
        # Please check the data structure here.
        print(batch_data['output'])
    with open("mmfi_config.yaml", "w") as file:
        yaml.dump(dataset_config.model_dump(),file,default_flow_style=None)