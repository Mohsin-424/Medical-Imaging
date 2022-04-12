# -*- coding:utf-8 -*-


"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import h5py
import random
import numpy as np
import torch
from torch.utils import data

from utils import centered_resize


class HDF5Dataset(data.Dataset):
    """
    class that loads hdf5 dataset object
    """

    def __init__(self, set, file_path, input_size: int = 256, transforms=None, p=0.5):
        """
        Args:
        """
        self.file_path = file_path
        self.set = set
        self.input_size = input_size
        self.transforms = transforms
        self.data = self._load_data()
        self.p = p # probability to do data augmentation 
        with h5py.File(self.file_path, 'r') as f:
            self.num_classes = int(f.attrs['num_classes'])
            self.num_modalities = int(f.attrs['num_modalities'])

            self.key_list = list(f[self.set].keys())

    def _load_data(self):
        """ List of images from the HDF5 file and save them into a set list.

        :return
            list containing path to files for the corresponding set
        """

        def get_set_list(set_key, file):
            set_list = []
            f_set = file[set_key]
            for key in list(f_set.keys()):
                patient = f_set[key]
                img = patient['img']
                gt = patient['gt']

                assert img.shape[0:3] == gt.shape[0:3]
                for i in range(img.shape[0]):  # sampling axial slices
                    k = '{}/{}'.format(set_key, key)
                    set_list.append((k, i))
            return set_list

        with h5py.File(self.file_path, 'r') as f:
            set_list = get_set_list(self.set, f)
            if self.set == 'train':
                set_list = np.random.permutation(set_list)
        return set_list

    def _open_dataset(self):
        self.f = h5py.File(self.file_path, 'r')

    def __getitem__(self, index):
        """This method loads, transforms and returns slice corresponding to the
        corresponding index.
        :arg
            index: the index of the slice within patient data
        :return
            A tuple (input, target)

        """
        key, position = self.data[index]
        if not hasattr(self, 'f'):
            self._open_dataset()

        img_slice, gt_slice = self.get_data_slice(key, position, self.f)
        # Resize the image and the ground truth
        img_slice = centered_resize(img_slice, (self.input_size, self.input_size))
        gt_slice = centered_resize(gt_slice, (self.input_size, self.input_size))

        if len(img_slice.shape) < 3:
            img_slice = img_slice[..., None]

        random.seed(780)
        np.random.seed(780)
        if self.transforms is not None:
            for t in self.transforms:
                p = np.random.rand()
                if p < 0.5: 
                    #img_slice = t(img_slice, is_label=False)
                    #gt_slice = t(gt_slice, is_label=True)
                    img_slice, gt_slice = t(img_slice, gt_slice)

        img_slice = torch.from_numpy(img_slice)
        gt_slice = torch.from_numpy(gt_slice)

        # H, W C -> C, W, H
        img_slice = torch.swapaxes(img_slice, 0, -1)
        gt_slice = torch.swapaxes(gt_slice, 0, -1)
        # K, W, H -> W, H
        gt_slice = gt_slice.argmax(0)

        return img_slice, gt_slice

    def __len__(self):
        """
        return the length of the dadaist
        """
        return int(np.floor(len(self.data)))

    @staticmethod
    def get_data_slice(key, position, file):
        """
        Return one slice from the hdf3 file
        Args:
            key: key corresponding to each patient
            position: image and ground truth positions into patient's images
            file: the hdf5 dataset
        :return
            tuple corresponding to a slice and its corresponding ground truth
        """
        img = np.array(file['{}/img'.format(key)])
        gt = np.array(file['{}/gt'.format(key)])

        # sampling axial slices
        return img[int(position)], gt[int(position)]
