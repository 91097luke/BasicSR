import argparse
import cv2
import geopandas as gpd
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio as rio
import shutil
import sys
import torch
from pathlib import Path
from rasterio.mask import mask
from shapely.geometry import Polygon, box

sys.path.append("/home/luke/code/BasicSR/")
from basicsr.data.paired_image_dataset import min_max_normalize, read_tiff, unnormalize
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.models.hat_model import HATModelTiff
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.options import dict2str, parse_options, yaml_load


def get_raster_polygon(raster_path):
    """
    Returns a GeoDataFrame containing the polygon of the raster's extent.
    """
    with rio.open(raster_path) as src:
        # Get the affine transform, CRS, width, and height
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height

        # Calculate the coordinates of the four corners
        top_left = transform * (0, 0)
        top_right = transform * (width, 0)
        bottom_right = transform * (width, height)
        bottom_left = transform * (0, height)

        # Create a Shapely Polygon from the corners
        raster_polygon = Polygon([top_left, top_right, bottom_right, bottom_left])

        # Create a GeoDataFrame with the polygon and its CRS
        gdf = gpd.GeoDataFrame(geometry=[raster_polygon], crs=crs)
    return gdf


def intersect_rasters(raster_path1, raster_path2):
    """
    Returns the intersection polygon of two rasters.
    """

    gdf1 = get_raster_polygon(raster_path1)
    gdf2 = get_raster_polygon(raster_path2)

    gdf2 = gdf2.to_crs(gdf1.crs)

    # Perform spatial intersection
    intersection = gpd.overlay(gdf1, gdf2, how="intersection")

    if intersection.empty:
        print("No intersection found between the two rasters.")
        return None
    else:
        minx, miny, maxx, maxy = intersection.total_bounds

        polygon = box(minx, miny, maxx, maxy)

        aoi_gdf_square = gpd.GeoDataFrame(geometry=[polygon], crs=intersection.crs)
        geometries = [geom for geom in aoi_gdf_square.geometry]

        cropped_raster_1, raster_1_transform = mask(
            dataset=rio.open(raster_path1), shapes=geometries, crop=True, filled=False
        )

        cropped_raster_2, raster_2_transform = mask(
            dataset=rio.open(raster_path2), shapes=geometries, crop=True, filled=False
        )

        raster_1_meta = rio.open(raster_path1).meta.copy()
        raster_2_meta = rio.open(raster_path2).meta.copy()

        raster_1_meta.update(
            {
                "height": cropped_raster_1.shape[1],
                "width": cropped_raster_1.shape[2],
                "transform": raster_1_transform,
            }
        )

        raster_2_meta.update(
            {
                "height": cropped_raster_2.shape[1],
                "width": cropped_raster_2.shape[2],
                "transform": raster_2_transform,
            }
        )

        raster_1_name = raster_path1.split(".")[0] + "_cropped.tif"
        raster_2_name = raster_path2.split(".")[0] + "_cropped.tif"

        if not os.path.exists(raster_1_name):
            with rio.open(raster_1_name, "w", **raster_1_meta) as dest:
                dest.write(cropped_raster_1)

        if not os.path.exists(raster_2_name):
            with rio.open(raster_2_name, "w", **raster_2_meta) as dest:
                dest.write(cropped_raster_2)

        return raster_1_name, raster_2_name


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
    new_transform = rio.transform.from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, img.shape[1], img.shape[0]
    )

    img = img.astype(np.uint16)
    meta.update(
        {
            "driver": meta["driver"] if "driver" in meta else "GTiff",
            "height": img.shape[0],
            "width": img.shape[1],
            "count": img.shape[2],
            "dtype": img.dtype,
            "crs": meta["crs"] if "crs" in meta else None,
            "transform": new_transform,
            "nodata": meta["nodata"] if "nodata" in meta else None,
        }
    )

    with rio.open(save_img_path, "w", **meta) as dst:
        for i in range(img.shape[2]):
            dst.write(img[:, :, i], i + 1)


def norm_band(band, lower, upper):
    """
    Normalize a band to the range [0, 1] based on the lower and upper percentiles.
    """
    band = np.clip(band, lower, upper)
    band = (band - band.min()) / (band.max() - band.min())
    return band


def enhance_contrast_per_band(img_data, lower_percentile=1.0, upper_percentile=99.0):

    img_data_stretched = np.zeros_like(img_data)
    for i, band in enumerate(img_data):
        band_ = np.where(band == 0, np.inf, band)
        lower = np.percentile(band_, lower_percentile)
        upper = np.percentile(band, upper_percentile)
        img_data_stretched[i] = norm_band(band, lower, upper) * 255

    return img_data_stretched.astype(np.uint8)


def inference(raster_path, min_max_file, opt, save_to):

    opt = yaml_load(opt)
    opt["is_train"] = False
    opt["dist"] = False

    model = HATModelTiff(opt)
    print("Model initialized successfully.")

    with open(min_max_file, "r") as f:
        min_max_values = json.load(f)

    img_lq, meta, selected_bands = read_tiff(
        raster_path,
        bands="bgr",
        min_values=min_max_values["min_values"],
        max_values=min_max_values["max_values"],
    )

    data = {"lq": img2tensor(img_lq).unsqueeze(0)}
    model.feed_data(data)
    model.pre_process()
    model.tile_process()
    model.post_process()

    visuals = model.get_current_visuals()

    with rio.open(raster_path) as dataset:
        bounds = dataset.bounds
    sr_img = convert2tif(
        [visuals["result"]], min_max_values["min_values"], min_max_values["max_values"]
    )
    save_tiff(sr_img, save_to, meta, bounds)


def calculate_area_psnr_ssim(
    gt_raster_path, lr_rater_path, sr_raster_path, min_max_file_gt, min_max_file_lq
):
    with open(min_max_file_gt, "r") as f:
        min_max_values_gt = json.load(f)

    with open(min_max_file_lq, "r") as f:
        min_max_values_lq = json.load(f)

    gt_raster_path, lr_rater_path = intersect_rasters(gt_raster_path, lr_rater_path)
    sr_raster_path, _ = intersect_rasters(sr_raster_path, lr_rater_path)

    gt_img, _, _ = read_tiff(
        gt_raster_path,
        bands="bgr",
        min_values=min_max_values_gt["min_values"],
        max_values=min_max_values_gt["max_values"],
    )

    lq_img, _, _ = read_tiff(
        lr_rater_path,
        bands="bgr",
        min_values=min_max_values_lq["min_values"],
        max_values=min_max_values_lq["max_values"],
    )

    sr_img, _, _ = read_tiff(
        sr_raster_path,
        bands="bgr",
        min_values=min_max_values_lq["min_values"],
        max_values=min_max_values_lq["max_values"],
    )

    lq_img = cv2.resize(
        lq_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )
    sr_img = cv2.resize(
        sr_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )

    psnr = calculate_psnr(sr_img * 255, gt_img * 255, crop_border=4)
    ssim = calculate_ssim(sr_img * 255, gt_img * 255, crop_border=4)

    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.4f}")


def get_planet_bgrnir(raster):
    raster_name = raster.split(".")[0] + "_bgrnir.tif"
    with rio.open(raster) as dataset:

        patch = dataset.read()[[1, 3, 5, 7], :, :]  # get bgrnir bands

        # You can then write out a new file
        meta = dataset.meta
        meta.update({"count": patch.shape[0]})

        with rio.open(raster_name, "w", **meta) as dst:
            dst.write(patch)

        dataset.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to options YAML file."
    )
    parser.add_argument(
        "-lr_raster_path", type=str, required=True, help="Path to input raster file."
    )
    parser.add_argument(
        "-gt_raster_path", type=str, required=False, help="Path to input raster file."
    )
    parser.add_argument(
        "-lr_min_max_file",
        type=str,
        default=None,
        help="Path to min-max values JSON file.",
    )
    parser.add_argument(
        "-gt_min_max_file",
        type=str,
        default=None,
        required=False,
        help="Path to min-max values JSON file.",
    )
    parser.add_argument(
        "-save_to", type=str, required=True, help="Path to save output raster file."
    )
    args = parser.parse_args()

    inference(args.lr_raster_path, args.lr_min_max_file, args.opt, args.save_to)
    if args.gt_raster_path is not None:
        calculate_area_psnr_ssim(
            args.gt_raster_path,
            args.lr_raster_path,
            args.save_to,
            args.gt_min_max_file,
            args.lr_min_max_file,
        )


def okion_experiments():
    aois = [
        "oikon_TC1_01",
        "oikon_TC1_02",
        "oikon_TC2_01",
        "oikon_TC2_02",
        "oikon_TC3_01",
        "oikon_TC3_02",
        "oikon_TC4_01",
        "oikon_TC4_02",
        "oikon_TC5_01",
        "oikon_TC5_02",
    ]

    for aoi in aois:
        src_dir = f"/home/luke/code/semablu-datasets/assets/{aoi}/lr_assets/CROPPED"

        raster_path = [
            raster_path for raster_path in Path(src_dir).rglob(f"*_B02_B03_B04_*.jp2")
        ][0]

        tile_name = raster_path.stem + "_SRx3.tif"
        save_to = f"/home/luke/code/semablu-datasets/assets/{aoi}/sr_assets/"
        os.makedirs(save_to, exist_ok=True)
        inference(
            raster_path=str(raster_path),
            min_max_file="/home/luke/code/BasicSR/datasets/potamia_dataset/lr/tif/min_max_values.json",
            opt="/home/luke/code/BasicSR/options/test/HAT/HAT-L_SRx4_ImageNet-pretrain_tiff.yml",
            save_to=f"{save_to}/{tile_name}",
        )


def malta_aois():
    aois = [
        # "/home/luke/code/semablu-datasets/assets/malta_20250705",
        "/home/luke/code/semablu-datasets/assets/malta_20250725",
    ]

    for aoi in aois:
        save_to = f"{aoi}/sr_assets/"
        os.makedirs(save_to, exist_ok=True)
        raster_paths = [raster_path for raster_path in Path(os.path.join(aoi, 'CROPPED')).rglob(f"*.tif")]

        for raster_path in raster_paths:
            tile_name = raster_path.stem + "_SRx3.tif"
            inference(
                raster_path=str(raster_path),
                min_max_file="/home/luke/code/BasicSR/datasets/potamia_dataset/lr/tif/min_max_values.json",
                opt="/home/luke/code/BasicSR/options/test/HAT/HAT-L_SRx4_ImageNet-pretrain_tiff.yml",
                save_to=f"{save_to}/{tile_name}",
            )


if __name__ == "__main__":
    malta_aois()
    # main()
    # python /home/luke/code/BasicSR/inference/inference_HAT.py -opt /home/luke/code/BasicSR/options/test/HAT/HAT-L_SRx4_ImageNet-pretrain_tiff.yml -lr_raster_path /home/luke/code/BasicSR/examples/zagreb/zagreb_lr.tif -gt_raster_path /home/luke/code/BasicSR/examples/zagreb/zagreb_hr_bgrnir.tif -save_to /home/luke/code/BasicSR/examples/zagreb/zagreb_sr.tif -lr_min_max_file /home/luke/code/BasicSR/datasets/potamia_dataset/lr/tif/min_max_values.json -gt_min_max_file /home/luke/code/BasicSR/datasets/potamia_dataset/hr/tif/min_max_values.json
