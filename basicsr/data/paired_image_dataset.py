
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file
)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

from os import path as osp
import random
import numpy as np
import torch


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration with optional Superpixel maps."""

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()

        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt.get('task', None)
        self.noise = opt.get('noise', 0)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']

        # Optional Superpixel map folder
        self.sp_folder = opt.get('dataroot_sp', None)

        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # --------------------------------------------------
        # Build paired image paths
        # --------------------------------------------------

        if self.io_backend_opt['type'] == 'lmdb':

            self.io_backend_opt['db_paths'] = [
                self.lq_folder,
                self.gt_folder
            ]

            self.io_backend_opt['client_keys'] = ['lq', 'gt']

            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt']
            )

        elif (
            'meta_info_file' in self.opt
            and self.opt['meta_info_file'] is not None
        ):

            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                self.opt['meta_info_file'],
                self.filename_tmpl
            )

        else:

            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder],
                ['lq', 'gt'],
                self.filename_tmpl,
                self.task
            )

    def __getitem__(self, index):

        # --------------------------------------------------
        # Initialize file client
        # --------------------------------------------------

        if self.file_client is None:

            self.file_client = FileClient(
                self.io_backend_opt.pop('type'),
                **self.io_backend_opt
            )

        scale = self.opt['scale']

        # --------------------------------------------------
        # Load GT and LQ images
        # --------------------------------------------------

        if self.task == 'CAR':

            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')

            img_gt = imfrombytes(
                img_bytes,
                flag='grayscale',
                float32=False
            )

            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')

            img_lq = imfrombytes(
                img_bytes,
                flag='grayscale',
                float32=False
            )

            img_gt = np.expand_dims(
                img_gt, axis=2
            ).astype(np.float32) / 255.

            img_lq = np.expand_dims(
                img_lq, axis=2
            ).astype(np.float32) / 255.

        elif self.task == 'denoising_gray':

            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path

            img_bytes = self.file_client.get(gt_path, 'gt')

            img_gt = imfrombytes(
                img_bytes,
                flag='grayscale',
                float32=True
            )

            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)

            img_lq = img_gt + np.random.normal(
                0,
                self.noise / 255.,
                img_gt.shape
            )

            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)

        elif self.task == 'denoising_color':

            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path

            img_bytes = self.file_client.get(gt_path, 'gt')

            img_gt = imfrombytes(
                img_bytes,
                float32=True
            )

            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)

            img_lq = img_gt + np.random.normal(
                0,
                self.noise / 255.,
                img_gt.shape
            )

        else:

            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')

            img_gt = imfrombytes(
                img_bytes,
                float32=True
            )

            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')

            img_lq = imfrombytes(
                img_bytes,
                float32=True
            )

        # --------------------------------------------------
        # Load corresponding Superpixel map
        # --------------------------------------------------

        sp_map = None

        if self.sp_folder is not None:

            sp_path = osp.join(
                self.sp_folder,
                osp.splitext(osp.basename(lq_path))[0] + '.npy'
            )

            if not osp.isfile(sp_path):
                raise FileNotFoundError(
                    f"Superpixel map not found: {sp_path}"
                )

            sp_map = np.load(sp_path)

            # Remove optional batch/channel dimensions
            sp_map = np.squeeze(sp_map)

            if sp_map.ndim != 2:
                raise ValueError(
                    f"Expected a 2D SP map, got shape {sp_map.shape}"
                )

            if np.any(sp_map < 0):
                raise ValueError(
                    f"Negative Superpixel IDs found in {sp_path}"
                )

            sp_map = sp_map.astype(np.int64)

            if sp_map.shape != img_lq.shape[:2]:
                raise ValueError(
                    f"SP map shape {sp_map.shape} "
                    f"does not match LR image shape {img_lq.shape[:2]}"
                )

        # --------------------------------------------------
        # Training augmentation
        # --------------------------------------------------

        if self.opt['phase'] == 'train':

            gt_size = self.opt['gt_size']

            if sp_map is not None:

                # LR crop size
                lq_size = gt_size // scale

                h_lq, w_lq = img_lq.shape[:2]

                if h_lq < lq_size or w_lq < lq_size:
                    raise ValueError(
                        f"LR image size {(h_lq, w_lq)} "
                        f"is smaller than crop size {lq_size}"
                    )

                if img_gt.shape[0] != h_lq * scale:
                    raise ValueError(
                        f"GT height {img_gt.shape[0]} "
                        f"does not match LR height {h_lq} * scale {scale}"
                    )

                if img_gt.shape[1] != w_lq * scale:
                    raise ValueError(
                        f"GT width {img_gt.shape[1]} "
                        f"does not match LR width {w_lq} * scale {scale}"
                    )

                # Random crop coordinates in LR space
                top = random.randint(0, h_lq - lq_size)
                left = random.randint(0, w_lq - lq_size)

                # Crop GT using scaled coordinates
                img_gt = img_gt[
                    top * scale:(top + lq_size) * scale,
                    left * scale:(left + lq_size) * scale,
                    :
                ]

                # Crop LQ
                img_lq = img_lq[
                    top:top + lq_size,
                    left:left + lq_size,
                    :
                ]

                # Crop Superpixel map using identical LR coordinates
                sp_map = sp_map[
                    top:top + lq_size,
                    left:left + lq_size
                ]

                # Add channel dimension for BasicSR augmentation
                sp_map = sp_map[:, :, None]

                # OpenCV does not support int64 arrays in cv2.flip.
                # Use int32 temporarily for augmentation.
                sp_map = sp_map.astype(np.int32)

                # Ensure all arrays are contiguous in memory.
                # This is important because BasicSR augmentation
                # performs in-place OpenCV flip operations.
                img_gt = np.ascontiguousarray(img_gt)
                img_lq = np.ascontiguousarray(img_lq)
                sp_map = np.ascontiguousarray(sp_map)

                # Apply exactly the same random augmentation
                # to GT, LQ and Superpixel map.
                img_gt, img_lq, sp_map = augment(
                    [img_gt, img_lq, sp_map],
                    self.opt['use_hflip'],
                    self.opt['use_rot']
                )

                # Remove channel dimension and restore int64
                sp_map = np.ascontiguousarray(
                    sp_map[:, :, 0],
                    dtype=np.int64
                )

            else:

                # Original BasicSR paired crop
                img_gt, img_lq = paired_random_crop(
                    img_gt,
                    img_lq,
                    gt_size,
                    scale,
                    gt_path
                )

                # Original BasicSR augmentation
                img_gt, img_lq = augment(
                    [img_gt, img_lq],
                    self.opt['use_hflip'],
                    self.opt['use_rot']
                )

        # --------------------------------------------------
        # Color space transformation
        # --------------------------------------------------

        if 'color' in self.opt and self.opt['color'] == 'y':

            img_gt = rgb2ycbcr(
                img_gt,
                y_only=True
            )[..., None]

            img_lq = rgb2ycbcr(
                img_lq,
                y_only=True
            )[..., None]

        # --------------------------------------------------
        # Crop unmatched GT during validation/testing
        # --------------------------------------------------

        if self.opt['phase'] != 'train':

            img_gt = img_gt[
                0:img_lq.shape[0] * scale,
                0:img_lq.shape[1] * scale,
                :
            ]

        # --------------------------------------------------
        # Convert images to tensors
        # --------------------------------------------------

        img_gt, img_lq = img2tensor(
            [img_gt, img_lq],
            bgr2rgb=True,
            float32=True
        )

        if self.mean is not None or self.std is not None:

            normalize(
                img_lq,
                self.mean,
                self.std,
                inplace=True
            )

            normalize(
                img_gt,
                self.mean,
                self.std,
                inplace=True
            )

        # --------------------------------------------------
        # Prepare output dictionary
        # --------------------------------------------------

        result = {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

        if sp_map is not None:

            result['sp_map'] = torch.from_numpy(
                np.ascontiguousarray(sp_map)
            ).long()

        return result

    def __len__(self):
        return len(self.paths)
