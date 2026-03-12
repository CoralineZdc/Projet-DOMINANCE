''' Fer2013 Dataset class'''

from __future__ import print_function

import torch
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data
import pandas as pd


class FER2013(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set

        if self.split == 'Training':
            self.data = pd.read_csv("./data/train-20240123-14902.csv")
            pixels_series = self.data['pixels']
            valence_series = self.data['Valence']

            processed_pixel_arrays = []
            processed_labels = []

            for idx, pixel_entry in enumerate(pixels_series):
                pixel_str = str(pixel_entry).strip()
                if pixel_str.lower() == 'nan' or not pixel_str:
                    continue
                try:
                    pixels = list(map(int, pixel_str.split()))
                    if len(pixels) == 48 * 48:
                        arr_2d = np.array(pixels, dtype=np.uint8).reshape(48, 48)
                        arr_3d = np.concatenate((arr_2d[:, :, np.newaxis], arr_2d[:, :, np.newaxis], arr_2d[:, :, np.newaxis]), axis=2)

                        image = Image.fromarray(arr_3d)
                        if self.transform is not None:
                            image = self.transform(image)

                        processed_pixel_arrays.append(image)
                        processed_labels.append(valence_series.iloc[idx])
                    else:
                        print(f"Warning: Training pixel string has incorrect length ({len(pixels)}) for 48x48 image at index {idx}. Skipping this entry.")
                except ValueError:
                    print(f"Warning: Could not parse training pixel string '{pixel_str}' at index {idx}. Skipping this entry.")
                    continue

            self.train_data = processed_pixel_arrays
            self.train_labels = torch.tensor(processed_labels, dtype=torch.float32)

        elif self.split == 'PublicTest':
            self.data = pd.read_csv("./data/publictest-20240507.csv")
            pixels_series = self.data['pixels']
            valence_series = self.data['Valence']

            processed_pixel_arrays = []
            processed_labels = []

            for idx, pixel_entry in enumerate(pixels_series):
                pixel_str = str(pixel_entry).strip()
                if pixel_str.lower() == 'nan' or not pixel_str:
                    continue
                try:
                    pixels = list(map(int, pixel_str.split()))
                    if len(pixels) == 48 * 48:
                        arr_2d = np.array(pixels, dtype=np.uint8).reshape(48, 48)
                        arr_3d = np.concatenate((arr_2d[:, :, np.newaxis], arr_2d[:, :, np.newaxis], arr_2d[:, :, np.newaxis]), axis=2)

                        image = Image.fromarray(arr_3d)
                        if self.transform is not None:
                            image = self.transform(image)

                        processed_pixel_arrays.append(image)
                        processed_labels.append(valence_series.iloc[idx])
                    else:
                        print(f"Warning: PublicTest pixel string has incorrect length ({len(pixels)}) for 48x48 image at index {idx}. Skipping this entry.")
                except ValueError:
                    print(f"Warning: Could not parse PublicTest pixel string '{pixel_str}' at index {idx}. Skipping this entry.")
                    continue

            self.PublicTest_data = processed_pixel_arrays
            self.PublicTest_labels = torch.tensor(processed_labels, dtype=torch.float32)

        else: # self.split == 'PrivateTest'
            self.data = pd.read_csv("./data/privatetest-20240506-yh.csv")
            pixels_series = self.data['pixels']
            valence_series = self.data['Valence']

            processed_pixel_arrays = []
            processed_labels = []

            for idx, pixel_entry in enumerate(pixels_series):
                pixel_str = str(pixel_entry).strip()
                if pixel_str.lower() == 'nan' or not pixel_str:
                    continue
                try:
                    pixels = list(map(int, pixel_str.split()))
                    if len(pixels) == 48 * 48:
                        arr_2d = np.array(pixels, dtype=np.uint8).reshape(48, 48)
                        arr_3d = np.concatenate((arr_2d[:, :, np.newaxis], arr_2d[:, :, np.newaxis], arr_2d[:, :, np.newaxis]), axis=2)

                        image = Image.fromarray(arr_3d)
                        if self.transform is not None:
                            image = self.transform(image)

                        processed_pixel_arrays.append(image)
                        processed_labels.append(valence_series.iloc[idx])
                    else:
                        print(f"Warning: PrivateTest pixel string has incorrect length ({len(pixels)}) for 48x48 image at index {idx}. Skipping this entry.")
                except ValueError:
                    print(f"Warning: Could not parse PrivateTest pixel string '{pixel_str}' at index {idx}. Skipping this entry.")
                    continue

            self.PrivateTest_data = processed_pixel_arrays
            self.PrivateTest_labels = torch.tensor(processed_labels, dtype=torch.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
