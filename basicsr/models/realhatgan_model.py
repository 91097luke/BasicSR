import numpy as np
import os
import os.path as osp
import random
import rasterio as rio
import torch
from collections import OrderedDict
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.data.degradations import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)
from basicsr.data.transforms import paired_random_crop
from basicsr.metrics import calculate_metric
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp, imwrite
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class RealHATGANModel(SRGANModel):
    """GAN-based Real_HAT Model.

    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RealHATGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(
            differentiable=False
        ).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get("queue_size", 180)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images."""
        data_type = random.choices(["real", "synth"], self.opt["data_proba"])[0]

        if data_type == "synth":
            if self.is_train and self.opt.get("high_order_degradation", True):
                # training data synthesis
                self.gt = data["gt"].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

                self.kernel1 = data["kernel1"].to(self.device)
                self.kernel2 = data["kernel2"].to(self.device)
                self.sinc_kernel = data["sinc_kernel"].to(self.device)

                ori_h, ori_w = self.gt.size()[2:4]

                # ----------------------- The first degradation process ----------------------- #
                # blur
                out = filter2D(self.gt_usm, self.kernel1)
                # random resize
                updown_type = random.choices(
                    ["up", "down", "keep"], self.opt["resize_prob"]
                )[0]
                if updown_type == "up":
                    scale = np.random.uniform(1, self.opt["resize_range"][1])
                elif updown_type == "down":
                    scale = np.random.uniform(self.opt["resize_range"][0], 1)
                else:
                    scale = 1
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                # add noise
                gray_noise_prob = self.opt["gray_noise_prob"]
                if np.random.uniform() < self.opt["gaussian_noise_prob"]:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.opt["noise_range"],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                    )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt["poisson_scale_range"],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                    )
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["jpeg_range"])
                out = torch.clamp(
                    out, 0, 1
                )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
                out = self.jpeger(out, quality=jpeg_p)

                # ----------------------- The second degradation process ----------------------- #
                # blur
                if np.random.uniform() < self.opt["second_blur_prob"]:
                    out = filter2D(out, self.kernel2)
                # random resize
                updown_type = random.choices(
                    ["up", "down", "keep"], self.opt["resize_prob2"]
                )[0]
                if updown_type == "up":
                    scale = np.random.uniform(1, self.opt["resize_range2"][1])
                elif updown_type == "down":
                    scale = np.random.uniform(self.opt["resize_range2"][0], 1)
                else:
                    scale = 1
                mode = random.choice(["area", "bilinear", "bicubic"])
                out = F.interpolate(
                    out,
                    size=(
                        int(ori_h / self.opt["scale"] * scale),
                        int(ori_w / self.opt["scale"] * scale),
                    ),
                    mode=mode,
                )
                # add noise
                gray_noise_prob = self.opt["gray_noise_prob2"]
                if np.random.uniform() < self.opt["gaussian_noise_prob2"]:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.opt["noise_range2"],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                    )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt["poisson_scale_range2"],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                    )

                # JPEG compression + the final sinc filter
                # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
                # as one operation.
                # We consider two orders:
                #   1. [resize back + sinc filter] + JPEG compression
                #   2. JPEG compression + [resize back + sinc filter]
                # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
                if np.random.uniform() < 0.5:
                    # resize back + the final sinc filter
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(
                        out,
                        size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                        mode=mode,
                    )
                    out = filter2D(out, self.sinc_kernel)
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(
                        *self.opt["jpeg_range2"]
                    )
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                else:
                    # JPEG compression
                    jpeg_p = out.new_zeros(out.size(0)).uniform_(
                        *self.opt["jpeg_range2"]
                    )
                    out = torch.clamp(out, 0, 1)
                    out = self.jpeger(out, quality=jpeg_p)
                    # resize back + the final sinc filter
                    mode = random.choice(["area", "bilinear", "bicubic"])
                    out = F.interpolate(
                        out,
                        size=(ori_h // self.opt["scale"], ori_w // self.opt["scale"]),
                        mode=mode,
                    )
                    out = filter2D(out, self.sinc_kernel)

                # clamp and round
                self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

                # random crop
                gt_size = self.opt["gt_size"]
                (self.gt, self.gt_usm), self.lq = paired_random_crop(
                    [self.gt, self.gt_usm], self.lq, gt_size, self.opt["scale"]
                )

                # training pair pool
                self._dequeue_and_enqueue()
                # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
                self.gt_usm = self.usm_sharpener(self.gt)
                self.lq = (
                    self.lq.contiguous()
                )  # for the warning: grad and param do not obey the gradient layout contract
            else:
                # for paired training or validation
                self.lq = data["lq"].to(self.device)
                if "gt" in data:
                    self.gt = data["gt"].to(self.device)
                    self.gt_usm = self.usm_sharpener(self.gt)

        elif data_type == "real":
            self.lq = data["lq"].to(self.device)
            if "gt" in data:
                self.gt = data["gt"].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RealHATGANModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img
        )
        self.is_train = True

    def optimize_parameters(self, current_iter):
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt["l1_gt_usm"] is False:
            l1_gt = self.gt
        if self.opt["percep_gt_usm"] is False:
            percep_gt = self.gt
        if self.opt["gan_gt_usm"] is False:
            gan_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (
            current_iter % self.net_d_iters == 0
            and current_iter > self.net_d_init_iters
        ):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict["l_g_pix"] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict["l_g_percep"] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict["l_g_style"] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict["l_g_gan"] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict["l_d_real"] = l_d_real
        loss_dict["out_d_real"] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict["l_d_fake"] = l_d_fake
        loss_dict["out_d_fake"] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt["network_g"]["window_size"]
        scale = self.opt.get("scale", 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[
            :, :, 0 : h - mod_pad_h * scale, 0 : w - mod_pad_w * scale
        ]


@MODEL_REGISTRY.register()
class RealHATGANModelTif(RealHATGANModel):

    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt["network_g"]["window_size"]
        self.scale = self.opt.get("scale", 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    def process(self):
        # model inference
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt["tile"]["tile_size"])
        tiles_y = math.ceil(height / self.opt["tile"]["tile_size"])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt["tile"]["tile_size"]
                ofs_y = y * self.opt["tile"]["tile_size"]
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt["tile"]["tile_size"], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt["tile"]["tile_size"], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt["tile"]["tile_pad"], 0)
                input_end_x_pad = min(input_end_x + self.opt["tile"]["tile_pad"], width)
                input_start_y_pad = max(input_start_y - self.opt["tile"]["tile_pad"], 0)
                input_end_y_pad = min(
                    input_end_y + self.opt["tile"]["tile_pad"], height
                )

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[
                    :,
                    :,
                    input_start_y_pad:input_end_y_pad,
                    input_start_x_pad:input_end_x_pad,
                ]

                # upscale tile
                try:
                    if hasattr(self, "net_g_ema"):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print("Error", error)
                print(f"\tTile {tile_idx}/{tiles_x * tiles_y}")

                # output tile area on total image
                output_start_x = input_start_x * self.opt["scale"]
                output_end_x = input_end_x * self.opt["scale"]
                output_start_y = input_start_y * self.opt["scale"]
                output_end_y = input_end_y * self.opt["scale"]

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt[
                    "scale"
                ]
                output_end_x_tile = (
                    output_start_x_tile + input_tile_width * self.opt["scale"]
                )
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt[
                    "scale"
                ]
                output_end_y_tile = (
                    output_start_y_tile + input_tile_height * self.opt["scale"]
                )

                # put tile into output image
                self.output[
                    :, :, output_start_y:output_end_y, output_start_x:output_end_x
                ] = output_tile[
                    :,
                    :,
                    output_start_y_tile:output_end_y_tile,
                    output_start_x_tile:output_end_x_tile,
                ]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[
            :,
            :,
            0 : h - self.mod_pad_h * self.scale,
            0 : w - self.mod_pad_w * self.scale,
        ]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        save_format = self.opt["val"].get("save_format")
        if save_format not in ["png", "tif"]:
            raise ValueError(
                f"Wrong save_format {save_format}. Supported ones are png and tif."
            )

        if with_metrics:
            if not hasattr(self, "metric_results"):  # only execute in the first run
                self.metric_results = {
                    metric: 0 for metric in self.opt["val"]["metrics"].keys()
                }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)

            self.pre_process()
            if "tile" in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()

            if save_format == "png":
                sr_img = dataloader.dataset.convert2img([visuals["result"]])
            elif save_format == "tif":
                sr_img = dataloader.dataset.convert2tif(
                    [visuals["result"]], val_data["native_scale"]
                )
            else:
                raise ValueError(
                    f"Wrong save_format {save_format}. Supported ones are png and tif."
                )

            metric_data["img"] = sr_img
            if "gt" in visuals:
                if save_format == "png":
                    gt_img = dataloader.dataset.convert2img([visuals["gt"]])
                elif save_format == "tif":
                    gt_img = dataloader.dataset.convert2tif(
                        [visuals["gt"]], val_data["native_scale"]
                    )
                else:
                    raise ValueError(
                        f"Wrong save_format {save_format}. Supported ones are png and tif."
                    )

                metric_data["img2"] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt["is_train"]:
                    save_img_path = osp.join(
                        self.opt["path"]["visualization"],
                        img_name,
                        f"{img_name}_{current_iter}.{save_format}",
                    )
                else:
                    if self.opt["val"]["suffix"]:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.{save_format}',
                        )
                    else:
                        save_img_path = osp.join(
                            self.opt["path"]["visualization"],
                            dataset_name,
                            f'{img_name}_{self.opt["name"]}.{save_format}',
                        )

                if save_format == "png":
                    imwrite(sr_img, save_img_path)
                elif save_format == "tif":

                    dir_name = os.path.abspath(os.path.dirname(save_img_path))
                    os.makedirs(dir_name, exist_ok=True)

                    meta = val_data["meta_gt"]
                    meta.update(
                        {
                            "driver": (
                                meta["driver"][0] if "driver" in meta else "GTiff"
                            ),
                            "height": sr_img.shape[0],
                            "width": sr_img.shape[1],
                            "count": sr_img.shape[2],
                            "dtype": sr_img.dtype,
                            "crs": meta["crs"][0] if "crs" in meta else None,
                            "transform": (
                                meta["transform"] if "transform" in meta else None
                            ),
                            "nodata": (
                                meta["nodata"].numpy() if "nodata" in meta else None
                            ),
                        }
                    )

                    with rio.open(save_img_path, "w", **meta) as dst:
                        for i in range(sr_img.shape[2]):
                            dst.write(sr_img[:, :, i], i + 1)
                else:
                    raise ValueError(
                        f"Wrong save_format {save_format}. Supported ones are png and tif."
                    )

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt["val"]["metrics"].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                # update the best metric result
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
