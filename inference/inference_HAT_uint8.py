import numpy as np
import rasterio as rio
import sys
from pathlib import Path
import os

sys.path.append("/home/luke/code/BasicSR/")
from basicsr.models.hat_model import HATModel
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.options import parse_options


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


def full_tile_inference(raster_path, save_to=None):

    opt, _ = parse_options("", is_train=False)
    name = opt["name"]
    model = HATModel(opt)
    print("Model initialized successfully.")

    if save_to is None:
        save_to = "/".join(raster_path.split("/")[:-1])
    tile_name = raster_path.split("/")[-1].split(".")[0]

    with rio.open(raster_path) as dataset:
        img = dataset.read()
        img = np.clip(img, 0, 10000)
        img = enhance_contrast_per_band(
            img, lower_percentile=1.0, upper_percentile=99.0
        )
        img = np.transpose(img, (1, 2, 0))

        # img = img[:, :, ::-1]  # BGR to RGB

        data = {"lq": img2tensor(img / 255.0).unsqueeze(0)}
        model.feed_data(data)
        model.pre_process()
        model.tile_process()
        model.post_process()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])

        dst_crs = dataset.crs.to_proj4()
        meta = dataset.meta
        bounds = dataset.bounds
        new_transform = rio.transform.from_bounds(
            bounds.left,
            bounds.bottom,
            bounds.right,
            bounds.top,
            sr_img.shape[1],
            sr_img.shape[0],
        )
        meta.update(
            {
                "driver": "GTiff",
                "height": sr_img.shape[0],
                "width": sr_img.shape[1],
                "count": 3,
                "dtype": "uint8",
                "nodata": None,
                "crs": dst_crs,
                "transform": new_transform,
            }
        )

    with rio.open(
        f"{save_to}/{tile_name}_RGB_uint8_SRx4.tif",
        "w",
        **meta,
    ) as dst:
        dst.write(sr_img[:, :, 0], 1)  # Red channel
        dst.write(sr_img[:, :, 1], 2)  # Green channel
        dst.write(sr_img[:, :, 2], 3)  # Blue channel

    new_transform = rio.transform.from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, img.shape[1], img.shape[0]
    )
    meta.update(
        {
            "driver": "GTiff",
            "height": img.shape[0],
            "width": img.shape[1],
            "count": 3,
            "dtype": "uint8",
            "nodata": None,
            "crs": dst_crs,
            "transform": new_transform,
        }
    )

    # img = (img * 255).astype(np.uint8)
    with rio.open(
        f"{save_to}/{tile_name}_RGB_uint8.tif",
        "w",
        **meta,
    ) as dst:
        dst.write(img[:, :, 0], 1)  # Red channel
        dst.write(img[:, :, 1], 2)  # Green channel
        dst.write(img[:, :, 2], 3)  # Blue channel

def malta_aois():
    aois = [
        "/home/luke/code/semablu-datasets/assets/malta_20250705/CROPPED",
        "/home/luke/code/semablu-datasets/assets/malta_20250725/CROPPED",
    ]

    for aoi in aois:
        save_to = f"{aoi}/sr_assets/"
        os.makedirs(save_to, exist_ok=True)
        raster_paths = [raster_path for raster_path in Path(aoi).rglob(f"*.tif")]

        for raster_path in raster_paths:
            full_tile_inference(
                raster_path=str(raster_path),
                save_to=save_to
            )

if __name__ == "__main__":
    malta_aois()
    # aois = [
    #     "oikon_TC1_01",
    #     "oikon_TC1_02",
    #     "oikon_TC2_01",
    #     "oikon_TC2_02",
    #     "oikon_TC3_01",
    #     "oikon_TC3_02",
    #     "oikon_TC4_01",
    #     "oikon_TC4_02",
    #     "oikon_TC5_01",
    #     "oikon_TC5_02",
    # ]

    # for aoi in aois:
    #     src_dir = f"/home/luke/code/semablu-datasets/assets/{aoi}/lr_assets/CROPPED"

    #     raster_path = [raster_path for raster_path in Path(src_dir).rglob(f"*_B02_B03_B04_*.jp2")][0]

    #     save_to = f"/home/luke/code/semablu-datasets/assets/{aoi}/sr_assets/"
    #     os.makedirs(save_to, exist_ok=True)
    #     full_tile_inference(str(raster_path), save_to)


