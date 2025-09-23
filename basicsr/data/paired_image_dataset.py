from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor, tensor2img
from basicsr.utils.options import copy_opt_file
from basicsr.utils.registry import DATASET_REGISTRY

import rasterio
import numpy as np
import cv2
from skimage.exposure import match_histograms
from tqdm import tqdm
import os
import json

import random
from rasterio.transform import Affine

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

def center_crop(image: np.ndarray, target_size: int):
    height, width, channel = image.shape
    if height != target_size or width != target_size:
        start_x = (width - target_size) // 2
        start_y = (height - target_size) // 2
        return image[start_y:start_y + target_size, start_x:start_x + target_size, :]
    return image

def norm_band(band, lower, upper):
    """
    Normalize a band to the range [0, 1] based on the lower and upper percentiles.
    """
    band = np.clip(band, lower, upper)
    band = (band - band.min()) / (band.max() - band.min())
    return band

def enhance_contrast_per_band(img_data, lower_percentile=1.0, upper_percentile=99.0):
    _, _, channel = img_data.shape

    img_data_stretched = np.zeros_like(img_data).astype(np.float64)
    for i in range(channel):
        band = img_data[:, :, i]
        band_ = np.where(band == 0, np.inf, band)
        lower = np.percentile(band_, lower_percentile)
        upper = np.percentile(band, upper_percentile)
        img_data_stretched[:, :, i]= norm_band(band, lower, upper) * 255

    return img_data_stretched.astype(np.uint8)

def min_max_normalize(img_data, min_values, max_values):
    channel, _, _ = img_data.shape

    img_data_normalized = np.zeros_like(img_data).astype(np.float64)
    for i in range(channel):
        band = img_data[i, :, :]
        min_value = min_values[i]
        max_value = max_values[i]
        band = np.clip(band, min_value, max_value)
        img_data_normalized[i, :, :] = (band - min_value) / (max_value - min_value)  # * 255

    return img_data_normalized

def unnormalize(img_data, min_values, max_values):
    _, _, channel = img_data.shape
    img_data_unnormalized = np.zeros_like(img_data).astype(np.float64)
    for i in range(channel):
        band = img_data[:, :, i]
        min_value = min_values[i]
        max_value = max_values[i]
        img_data_unnormalized[:, :, i] = band * (max_value - min_value) + min_value  # * 255
    return img_data_unnormalized

def get_bands(image: np.ndarray, bands: str):
    """Select specific bands from the image.

    Args:
        image (np.ndarray): Input image with shape (H, W, C).
        bands (str): Bands to select. Options are 'bgr', 'bgnir', 'all'.
    Returns:
        np.ndarray: Image with selected bands.
    """

    if bands == 'bgr':
        return image[[0, 1, 2], :, :], 'bgr'
    elif bands == 'bgnir':
        return image[[0, 1, 3], :, :], 'bgnir'
    elif bands == 'all':
        if random.random() < 0.5:
            return image[[0, 1, 2], :, :], 'bgr' # bgr
        else:
            return image[[0, 1, 3], :, :], 'bgnir'  # bgnir
    else:
        raise ValueError(f'Unsupported bands: {bands}. Supported ones are "bgr", "bgnir", "all".')
def create_exponential_gradient(width, height, decay_rate=5.0):
    """
    Creates a 2D image with pixel values that exponentially increase from the
    edges to a maximum of 1 at the center.

    Args:
        width (int): The width of the desired image.
        height (int): The height of the desired image.
        decay_rate (float, optional): A positive value that controls the rate
                                      at which the pixel values decrease from the
                                      center. Higher values result in a sharper,
                                      more focused gradient. Defaults to 5.0.

    Returns:
        np.ndarray: A 2D NumPy array of shape (height, width) with float
                    pixel values between 0 and 1.
    """
    # Create coordinate grids for x and y
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Calculate the squared distance from the center for each pixel
    # The center is at (0, 0) in the normalized grid
    distance_squared = xx**2 + yy**2

    # Apply a negative exponential function to the distance
    # exp(-0) = 1 at the center, and values decrease exponentially outwards
    image = np.exp(-decay_rate * distance_squared)

    return np.expand_dims(image, axis=-1)  # Add channel dimension

def blended_cutmix(img_lq, img_lq2, img_gt, img_gt2, scale, alpha=0.5):
    """
    Performs blended CutMix data augmentation on two NumPy arrays (images).

    Blended CutMix works by creating a cutout region on img1 and replacing it
    with a blend of the corresponding region from img2. The blending is
    controlled by a mixing coefficient lambda.

    Args:
        img1 (np.ndarray): The first image, a NumPy array of shape (H, W, C).
        img2 (np.ndarray): The second image, a NumPy array of the same shape.
        alpha (float, optional): A hyperparameter for the beta distribution
                                 from which the mixing coefficient lambda is
                                 drawn. Defaults to 1.0, which corresponds to
                                 a uniform distribution.

    Returns:
        np.ndarray: The augmented image, a NumPy array of the same shape.
    """
    # Check if images have the same shape
    if img_lq.shape != img_lq2.shape:
        raise ValueError("Input images must have the same shape.")

    h, w, c = img_lq.shape
    buffer = 5

    # Draw a mixing coefficient lambda from a Beta distribution
    # A value of 1.0 for alpha results in a uniform distribution (standard CutMix)
    lam = np.random.beta(alpha, alpha)

    # Generate a random bounding box for the cutout region
    cut_w = np.random.uniform(low=buffer, high=w-buffer) * np.sqrt(1 - lam)
    cut_h = np.random.uniform(low=buffer, high=h-buffer) * np.sqrt(1 - lam)

    cx = np.random.uniform(low=buffer, high=w-buffer)
    cy = np.random.uniform(low=buffer, high=h-buffer)

    # Determine the coordinates of the bounding box
    x1 = np.clip(int(cx - cut_w / 2), 0, w)
    y1 = np.clip(int(cy - cut_h / 2), 0, h)
    x2 = np.clip(int(cx + cut_w / 2), 0, w)
    y2 = np.clip(int(cy + cut_h / 2), 0, h)

    # Scale the bounding box coordinates for the GT image
    x1_gt, y1_gt, x2_gt, y2_gt = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)

    # Create a copy of the first image to modify
    augmented_img_lq = np.copy(img_lq)
    augmented_img_gt = np.copy(img_gt)

    # Calculate the blend coefficient for the cutout region
    # The new lambda for blending is the original lambda plus a small value
    # to prevent a completely transparent blend.
    decay_rate = np.random.uniform(1.0, 2.0)
    exp_blend_coeff_lq = create_exponential_gradient(x2 - x1, y2 - y1, decay_rate)
    exp_blend_coeff_gt = create_exponential_gradient(x2_gt - x1_gt, y2_gt - y1_gt, decay_rate)

    # Blend the cutout regions of the two images
    augmented_img_lq[y1:y2, x1:x2, :] = (augmented_img_lq[y1:y2, x1:x2, :] * (1 - exp_blend_coeff_lq) +
                                      img_lq2[y1:y2, x1:x2, :] * exp_blend_coeff_lq)


    augmented_img_gt[y1_gt:y2_gt, x1_gt:x2_gt, :] = (augmented_img_gt[y1_gt:y2_gt, x1_gt:x2_gt, :] * (1 - exp_blend_coeff_gt) +
                                      img_gt2[y1_gt:y2_gt, x1_gt:x2_gt, :] * exp_blend_coeff_gt)

    return augmented_img_lq, augmented_img_gt

def mixup(img_lq, img_lq2, img_gt, img_gt2, alpha=1.0):
    """
    Performs Mixup data augmentation on two NumPy arrays (images).

    Mixup works by taking a weighted sum of two images. The blending ratio is
    determined by a random value drawn from a Beta distribution.

    Args:
        img1 (np.ndarray): The first image, a NumPy array of shape (H, W, C).
        img2 (np.ndarray): The second image, a NumPy array of the same shape.
        alpha (float, optional): A hyperparameter for the beta distribution
                                 from which the mixing coefficient lambda is
                                 drawn. Defaults to 1.0, which corresponds to
                                 a uniform distribution.

    Returns:
        np.ndarray: The augmented image, a NumPy array of the same shape.
    """
    # Check if images have the same shape
    if img_lq.shape != img_lq2.shape:
        raise ValueError("Input images must have the same shape.")

    if img_gt.shape != img_gt2.shape:
        raise ValueError("Input images must have the same shape.")

    # Draw a mixing coefficient lambda from a Beta distribution
    # A value of 1.0 for alpha results in a uniform distribution
    lam = np.random.beta(alpha, alpha)

    # Perform the linear combination of the two images
    augmented_img_lq = lam * img_lq + (1 - lam) * img_lq2
    augmented_img_gt = lam * img_gt + (1 - lam) * img_gt2

    return augmented_img_lq, augmented_img_gt

@DATASET_REGISTRY.register()
class TiffPairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(TiffPairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

        # get min max values for normalization
        root_path_gt = '/'.join(self.opt['dataroot_gt'].split('/')[:-1])
        root_path_lq = '/'.join(self.opt['dataroot_lq'].split('/')[:-1])

        self.min_values_gt, self.max_values_gt = self.get_dataset_min_max('gt', root_path_gt)
        self.min_values_lq, self.max_values_lq = self.get_dataset_min_max('lq', root_path_lq)


    def get_dataset_min_max(self, name, path):

        min_max_file = f'{path}/min_max_values.json'
        if os.path.exists(min_max_file):
            print(f'Loading {name} min max values from {min_max_file}')

            with open(min_max_file, 'r') as f:
                min_max_values = json.load(f)
            min_values = min_max_values['min_values']
            max_values = min_max_values['max_values']
            print(f'Loaded GT BGRNIR min values: {min_values}')
            print(f'Loaded GT BGRNIR max values: {max_values}')
            return min_values, max_values
        else:
            print(f'Calculating {name} min max values and saving to {min_max_file}')
            img_paths = [p['gt_path'] for p in self.paths] if name == 'gt' else [p['lq_path'] for p in self.paths]
            min_values, max_values = self.calculate_dataset_min_max(img_paths)
            print(f'Calculated {name} BGRNIR min values: {min_values}')
            print(f'Calculated {name} BGRNIR max values: {max_values}')

            min_max_values = {
                'min_values': min_values,
                'max_values': max_values
            }
            with open(min_max_file, 'w') as f:
                json.dump(min_max_values, f, indent=4)

            return min_values, max_values


    def calculate_dataset_min_max(self, img_paths):
        min_values = [np.inf, np.inf, np.inf, np.inf] # b g r nir
        max_values = [-np.inf, -np.inf, -np.inf, -np.inf]

        for path in tqdm(img_paths):
            with rasterio.open(path) as dataset:
                img_lq2 = dataset.read()
                for c in range(img_lq2.shape[0]):
                    band = img_lq2[c, :, :]
                    band_ = np.where(band == 0, np.inf, band)
                    min_value = np.min(band_)
                    max_value = np.max(band)

                    if min_value < min_values[c]:
                        min_values[c] = min_value.item()
                    if max_value > max_values[c]:
                        max_values[c] = max_value.item()
            dataset.close()
        return min_values, max_values

    def read_tiff(self, path, bands, min_values, max_values):
        with rasterio.open(path) as dataset:
            img = dataset.read()
            img = min_max_normalize(img, min_values, max_values)
            img, selected_bands = get_bands(img, bands)
            img = img.transpose(1, 2, 0)  # HWC
            meta = dataset.meta
        dataset.close()
        return img, meta, selected_bands

    def augment_2(self, img_gt, img_lq, index, bands, lq_size, native_scale, scale):

        if random.random() < 0.8:
            return img_gt, img_lq
        else:
                # randomly select another image in the dataset
                index2 = random.randint(0, len(self.paths) - 1)
                while index2 == index:
                    index2 = random.randint(0, len(self.paths) - 1)

                gt_path2 = self.paths[index2]['gt_path']
                lq_path2 = self.paths[index2]['lq_path']

                img_gt2,_ , _ = self.read_tiff(gt_path2, bands, self.min_values_gt, self.max_values_gt)
                img_lq2,_ , _ = self.read_tiff(lq_path2, bands, self.min_values_lq, self.max_values_lq)

                img_lq2 = center_crop(img_lq2, lq_size)
                img_gt2 = center_crop(img_gt2, int(np.round(lq_size * native_scale)))
                img_gt2 = cv2.resize(img_gt2, (lq_size * scale, lq_size * scale), interpolation=cv2.INTER_LANCZOS4)

                img_gt2 = match_histograms(img_gt2, img_lq2, channel_axis=-1)

                if self.opt['use_cutmix'] and self.opt['use_mixup']:
                    # put ratio in config
                    if random.random() < 0.5:
                        img_lq, img_gt = blended_cutmix(img_lq, img_lq2, img_gt, img_gt2, scale)
                    else:
                        img_lq, img_gt = mixup(img_lq, img_lq2, img_gt, img_gt2)
                elif self.opt['use_cutmix']:
                    img_lq, img_gt = blended_cutmix(img_lq, img_lq2, img_gt, img_gt2, scale)
                elif self.opt['use_mixup']:
                    img_lq, img_gt = mixup(img_lq, img_lq2, img_gt, img_gt2)
                else:
                    raise ValueError('Either use_cutmix or use_mixup must be True to augment with another image.')

                # add color jitter
                return img_gt, img_lq

    def convert2img(self, img_data):
        results = []
        imgs = tensor2img(img_data, min_max=(0, 1), out_type=np.float64)

        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        for img in imgs:
            img = unnormalize(img,self.min_values_lq, self.max_values_lq)
            img = enhance_contrast_per_band(img)
            results.append(img)
        if len(results) == 1:
            results = results[0]
        return results

    def update_metadata(self, meta, img):
        # Save the transform of the center-cropped raster for georeferencing
        # Compute the offset for the crop
        new_height, new_width = img.shape[0], img.shape[1]
        orig_height, orig_width = meta['height'], meta['width']
        start_x = (orig_width - new_width) // 2
        start_y = (orig_height - new_height) // 2

        # Update the transform
        orig_transform = meta['transform']
        new_transform = orig_transform * Affine.translation(start_x, start_y)
        meta['transform'] = new_transform
        meta['height'] = img.shape[0]
        meta['width'] = img.shape[1]
        meta['count'] = img.shape[2]  # Update the number of bands if necessary
        meta['crs'] = str(meta['crs'])
        return meta

    def convert2tif(self, img_data, native_scale):
        results = []
        imgs = tensor2img(img_data, min_max=(0, 1), out_type=np.float64)

        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        for img in imgs:
            img = unnormalize(img,self.min_values_lq, self.max_values_lq)
            img = cv2.resize(img,( int(np.round(self.opt['lq_size'] * native_scale)), int(np.round(self.opt['lq_size'] * native_scale))), interpolation=cv2.INTER_LANCZOS4)
            results.append(img)
        if len(results) == 1:
            results = results[0]
        return results


    def __getitem__(self, index):
        # if self.file_client is None:
        #     self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        bands = self.opt['bands'] if 'bands' in self.opt else 'bgr'  # default to bgr

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        img_gt, meta_gt, selected_bands = self.read_tiff(gt_path, bands, self.min_values_gt, self.max_values_gt)
        img_lq, _, _ = self.read_tiff(lq_path, selected_bands, self.min_values_lq, self.max_values_lq)

        lq_size = self.opt['lq_size']
        native_scale = img_gt.shape[0] / img_lq.shape[0]

        img_lq = center_crop(img_lq, lq_size)
        img_gt = center_crop(img_gt, int(np.round(lq_size * native_scale)))
        meta_gt = self.update_metadata(meta_gt, img_gt)

        img_gt = cv2.resize(img_gt, (lq_size * scale, lq_size * scale), interpolation=cv2.INTER_LANCZOS4)



        # enhance contrast per band
        img_gt = match_histograms(img_gt, img_lq, channel_axis=-1)

        # gt_vis = visualize_img(img_gt, self.min_values_gt, self.max_values_gt)
        # lq_vis = visualize_img(img_lq, self.min_values_lq, self.max_values_lq)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt.copy(), img_lq.copy()], self.opt['use_hflip'], self.opt['use_rot'])

            if self.opt['use_cutmix'] or self.opt['use_mixup']:
                img_gt, img_lq = self.augment_2(img_gt, img_lq, index, selected_bands, lq_size, native_scale, scale)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=False)
        # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'meta_gt': meta_gt, 'native_scale': native_scale}

    def __len__(self):
        return len(self.paths)