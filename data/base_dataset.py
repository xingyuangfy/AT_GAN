"""
Base dataset module for AT-GAN
Copyright (c) Xingyuangfy 2025. All rights reserved.

This module provides the base dataset class that defines the basic interface
for all dataset implementations in the project.
"""

import torch.utils.data as data


class BaseDataset(data.Dataset):
    """
    Base dataset class that provides basic dataset interface.
    
    This class serves as a template for all dataset implementations,
    ensuring consistent interface across different dataset types.
    """

    def __init__(self):
        """
        Initialize the base dataset.
        """
        super(BaseDataset, self).__init__()

    def name(self):
        """
        Return the name of the dataset.

        Returns:
            str: Name of the dataset
        """
        return "BaseDataset"

    def initialize(self, opt):
        """
        Initialize the dataset with given options.
        Must be implemented by subclasses.

        Args:
            opt: Configuration object containing dataset parameters
            
        Raises:
            NotImplementedError: If subclass does not implement this method
        """
        raise NotImplementedError("Subclass must implement the 'initialize' method.")