"""
Data loader module for AT-GAN
Copyright (c) Xingyuangfy 2025. All rights reserved.

This module provides data loading functionality for the aging GAN model,
handling batch processing and data iteration.
"""

import torch.utils.data
from data.multiclass_unaligned_dataset import MulticlassUnalignedDataset
from pdb import set_trace as st


class AgingDataLoader:
    """
    Data loader class for creating and managing PyTorch DataLoader.
    
    This class handles the loading and batching of aging-related image data,
    providing an interface for training and evaluation.
    """

    def __init__(self, opt):
        """
        Initialize AgingDataLoader.

        Args:
            opt: Configuration object containing dataloader parameters
        """
        self.opt = opt
        self.dataset = None
        self.dataloader = None
        self.initialize()

    def name(self):
        """
        Return the name of the data loader.

        Returns:
            str: Name of the data loader
        """
        return "AgingDataLoader"

    def initialize(self):
        """
        Initialize dataset and data loader.
        
        Creates the dataset instance and configures the PyTorch DataLoader
        with specified batch size and threading options.
        """
        self.dataset = self._create_dataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            drop_last=True,
            num_workers=int(self.opt.nThreads)
        )

    def load_data(self):
        """
        Return the data loader instance.

        Returns:
            DataLoader: PyTorch DataLoader instance
        """
        return self.dataloader

    def __len__(self):
        """
        Return the length of the dataset (limited to maximum dataset size).

        Returns:
            int: Length of the dataset
        """
        return min(len(self.dataset), self.opt.max_dataset_size)

    @staticmethod
    def _create_dataset():
        """
        Create and initialize the dataset.

        Returns:
            MulticlassUnalignedDataset: Initialized dataset instance
        """
        dataset = MulticlassUnalignedDataset()
        print(f"Dataset [{dataset.name()}] was created")
        dataset.initialize()
        return dataset


def create_data_loader(opt):
    """
    Create and return an AgingDataLoader instance.

    Args:
        opt: Configuration object containing dataloader parameters

    Returns:
        AgingDataLoader: Initialized data loader instance
    """
    data_loader = AgingDataLoader(opt)
    print(data_loader.name())
    return data_loader