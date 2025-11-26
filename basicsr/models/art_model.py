import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp
import rasterio as rio
import os


@MODEL_REGISTRY.register()
class ARTModel(SRModel):
    """ART model for image restoration."""

    # test by partitioning
    def test_tile(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class ARTModelTiff(ARTModel):

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        save_format = self.opt['val'].get('save_format')
        if save_format not in ['png', 'tif']:
            raise ValueError(f'Wrong save_format {save_format}. Supported ones are png and tif.')

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            self.test()
            visuals = self.get_current_visuals()

            if save_format == 'png':
                sr_img = dataloader.dataset.convert2img([visuals['result']])
            elif save_format == 'tif':
                sr_img = dataloader.dataset.convert2tif([visuals['result']], val_data['native_scale'])
            else:
                raise ValueError(f'Wrong save_format {save_format}. Supported ones are png and tif.')

            metric_data['img'] = sr_img
            if 'gt' in visuals:
                if save_format == 'png':
                    gt_img = dataloader.dataset.convert2img([visuals['gt']])
                elif save_format == 'tif':
                    gt_img = dataloader.dataset.convert2tif([visuals['gt']], val_data['native_scale'])
                else:
                    raise ValueError(f'Wrong save_format {save_format}. Supported ones are png and tif.')

                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.{save_format}')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.{save_format}')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.{save_format}')

                if save_format == 'png':
                    imwrite(sr_img, save_img_path)
                elif save_format == 'tif':

                    dir_name = os.path.abspath(os.path.dirname(save_img_path))
                    os.makedirs(dir_name, exist_ok=True)

                    meta = val_data['meta_gt']
                    meta.update({"driver": meta["driver"][0] if "driver" in meta else "GTiff",
                                 "height": sr_img.shape[0],
                                 "width": sr_img.shape[1],
                                 "count": sr_img.shape[2],
                                 "dtype": sr_img.dtype,
                                 "crs": meta["crs"][0] if "crs" in meta else None,
                                 "transform": meta["transform"] if "transform" in meta else None,
                                 "nodata": meta["nodata"].numpy() if "nodata" in meta else None
                                 })

                    with rio.open(save_img_path, 'w', **meta) as dst:
                        for i in range(sr_img.shape[2]):
                            dst.write(sr_img[:, :, i], i + 1)
                else:
                    raise ValueError(f'Wrong save_format {save_format}. Supported ones are png and tif.')

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)