import argparse
import cv2
import glob
import os
import shutil
import torch

import sys
import json
import numpy as np
import rasterio as rio

sys.path.append('/home/luke/code/BasicSR/')
from basicsr.models.hat_model import HATModelTiff
from basicsr.utils.options import dict2str, parse_options
from basicsr.data.paired_image_dataset import read_tiff, min_max_normalize, unnormalize
from basicsr.utils import tensor2img, img2tensor
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.options import yaml_load



def convert2tif(img_data, min_values, max_values):
    results = []
    imgs = tensor2img(img_data, min_max=(0, 1), out_type=np.float64)

    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    for img in imgs:
        img = unnormalize(img, min_values, max_values)
        results.append(img)

    if len(results) == 1:
        results = results[0]
    return results

def save_tiff(img, save_img_path, meta, bounds):
    new_transform = rio.transform.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, img.shape[1], img.shape[0])

    meta.update({"driver": meta["driver"] if "driver" in meta else "GTiff",
                "height": img.shape[0],
                "width": img.shape[1],
                "count": img.shape[2],
                "dtype": img.dtype,
                "crs": meta["crs"] if "crs" in meta else None,
                "transform": new_transform,
                "nodata": meta["nodata"] if "nodata" in meta else None
                })

    with rio.open(save_img_path, 'w', **meta) as dst:
        for i in range(img.shape[2]):
            dst.write(img[:, :, i], i + 1)

def inference(raster_path, min_max_file, opt, save_to):

    opt = yaml_load(opt)
    opt['is_train'] = False
    opt['dist'] = False
    
    model = HATModelTiff(opt)
    print("Model initialized successfully.")

    min_max_file = f'/home/luke/code/BasicSR/datasets/potamia_dataset/lr/tif/min_max_values.json'
    with open(min_max_file, 'r') as f:
        min_max_values = json.load(f)

    img_lq, meta, selected_bands = read_tiff(raster_path,
                                             bands='bgr',min_values=min_max_values['min_values'],
                                             max_values=min_max_values['max_values'])

    data = {'lq': img2tensor(img_lq).unsqueeze(0)}
    model.feed_data(data)
    model.pre_process()
    model.tile_process()
    model.post_process()

    visuals = model.get_current_visuals()

    with rio.open(raster_path) as dataset:
        bounds = dataset.bounds
    sr_img = convert2tif([visuals['result']], min_max_values['min_values'], min_max_values['max_values'])
    save_tiff(sr_img, save_to, meta, bounds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    parser.add_argument('-raster_path', type=str, required=True, help='Path to input raster file.')
    parser.add_argument('-min_max_file', type=str, default=None, help='Path to min-max values JSON file.')
    parser.add_argument('-save_to', type=str, required=True, help='Path to save output raster file.')
    args = parser.parse_args()

    inference(args.raster_path, args.min_max_file, args.opt, args.save_to)

if __name__ == '__main__':
    main()