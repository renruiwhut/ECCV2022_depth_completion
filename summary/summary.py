"""
    Reference from : https://github.com/zzangjinsun/NLSPN_ECCV20
"""


from . import BaseSummary
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

cm = plt.get_cmap('jet')


class Summary(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] " \
            "but got {}".format(mode)

        super(Summary, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def update(self, global_step, sample, output):
        output, uncertainty = output
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        dep = sample['dep'].detach().data.cpu().numpy()
        gt = sample['gt'].detach().data.cpu().numpy()
        pred = output.detach().data.cpu().numpy()
        uncertainty = uncertainty.detach().data.cpu().numpy()
        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

            rgb = rgb[0:num_summary, :, :, :]
            dep = dep[0:num_summary, :, :, :]
            gt = gt[0:num_summary, :, :, :]
            pred = pred[0:num_summary, :, :, :]
            uncertainty = uncertainty[0:num_summary, :, :, :]

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)


        list_img = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            dep_tmp = dep[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :]
            uncertainty_tmp = uncertainty[b, 0, :, :]
            max_depth_tmp = max(np.max(gt_tmp), np.max(pred_tmp))

            dep_tmp = 255.0 * dep_tmp / max_depth_tmp
            gt_tmp = 255.0 * gt_tmp / max_depth_tmp
            pred_tmp = 255.0 * pred_tmp / max_depth_tmp
            uncertainty_tmp = 255 * (uncertainty_tmp - uncertainty_tmp.min()) / (uncertainty_tmp.max() - uncertainty_tmp.min())

            dep_tmp = cm(dep_tmp.astype('uint8'))
            gt_tmp = cm(gt_tmp.astype('uint8'))
            pred_tmp = cm(pred_tmp.astype('uint8'))
            uncertainty_tmp = cm(uncertainty_tmp.astype('uint8'))
            dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            uncertainty_tmp = np.transpose(uncertainty_tmp[:, :, :3], (2, 0, 1))
            img = np.concatenate((rgb_tmp, dep_tmp, pred_tmp, gt_tmp, uncertainty_tmp), axis=1)
                                  
            list_img.append(img)

        img_total = np.concatenate(list_img, axis=2)
        img_total = torch.from_numpy(img_total)

        self.add_image(self.mode + '/images', img_total, global_step)

        # self.flush()

        # Reset
        self.loss = []
        self.metric = []

    def update_step2(self, global_step, sample, output_step1, output_step2):
        output, uncertainty = output_step1
        res, pred_step2 = output_step2
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:04d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:04d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        dep = sample['dep'].detach().data.cpu().numpy()
        gt = sample['gt'].detach().data.cpu().numpy()
        pred = output.detach().data.cpu().numpy()
        uncertainty = uncertainty.detach().data.cpu().numpy()
        res = res.detach().data.cpu().numpy()
        pred_step2 = pred_step2.detach().data.cpu().numpy()
        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

            rgb = rgb[0:num_summary, :, :, :]
            dep = dep[0:num_summary, :, :, :]
            gt = gt[0:num_summary, :, :, :]
            pred = pred[0:num_summary, :, :, :]
            uncertainty = uncertainty[0:num_summary, :, :, :]
            res = res[0:num_summary, :, :, :]
            pred_step2 = pred_step2[0:num_summary, :, :, :]

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)
        res = np.clip(res, a_min=0, a_max=self.args.max_depth)
        pred_step2 = np.clip(pred_step2, a_min=0, a_max=self.args.max_depth)

        list_img = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            dep_tmp = dep[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :]
            uncertainty_tmp = uncertainty[b, 0, :, :]
            res_tmp = res[b, 0, :, :]
            pred_step2_tmp = pred_step2[b, 0, :, :]
            max_depth_tmp = max(np.max(gt_tmp), np.max(pred_tmp))

            dep_tmp = 255.0 * dep_tmp / max_depth_tmp
            gt_tmp = 255.0 * gt_tmp / max_depth_tmp
            pred_tmp = 255.0 * pred_tmp / max_depth_tmp
            res_tmp = 255.0 * res_tmp / max_depth_tmp
            pred_step2_tmp = 255.0 * pred_step2_tmp / max_depth_tmp
            uncertainty_tmp = 255 * (uncertainty_tmp - uncertainty_tmp.min()) / (
                        uncertainty_tmp.max() - uncertainty_tmp.min())

            dep_tmp = cm(dep_tmp.astype('uint8'))
            gt_tmp = cm(gt_tmp.astype('uint8'))
            pred_tmp = cm(pred_tmp.astype('uint8'))
            uncertainty_tmp = cm(uncertainty_tmp.astype('uint8'))
            res_tmp = cm(res_tmp.astype('uint8'))
            pred_step2_tmp = cm(pred_step2_tmp.astype('uint8'))

            dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            res_tmp = np.transpose(res_tmp[:, :, :3], (2, 0, 1))
            pred_step2_tmp = np.transpose(pred_step2_tmp[:, :, :3], (2, 0, 1))
            uncertainty_tmp = np.transpose(uncertainty_tmp[:, :, :3], (2, 0, 1))
            img = np.concatenate((rgb_tmp, dep_tmp, pred_tmp, gt_tmp, uncertainty_tmp, res_tmp, pred_step2_tmp), axis=1)

            list_img.append(img)

        img_total = np.concatenate(list_img, axis=2)
        img_total = torch.from_numpy(img_total)

        self.add_image(self.mode + '/images', img_total, global_step)

        # self.flush()

        # Reset
        self.loss = []
        self.metric = []

    def save(self, epoch, idx, sample, output):
        with torch.no_grad():
            if self.args.save_result_only:
                self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir, self.mode, epoch)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_pred = '{}/{:010d}.png'.format(self.path_output, idx)

                pred = output.detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()

                pred = (pred*256.0).astype(np.uint16)
                pred = Image.fromarray(pred)
                pred.save(path_save_pred)
            else:
                # Parse data
                rgb = sample['rgb'].detach()
                dep = sample['dep'].detach()
                gt = sample['gt'].detach()
                output, uncertainty = output
                pred = output.detach()

                pred = torch.clamp(pred, min=0)

                # Un-normalization
                rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))

                rgb = rgb[0, :, :, :].data.cpu().numpy()
                dep = dep[0, 0, :, :].data.cpu().numpy()
                pred = pred[0, 0, :, :].data.cpu().numpy()
                gt = gt[0, 0, :, :].data.cpu().numpy()

                rgb = 255.0*np.transpose(rgb, (1, 2, 0))
                dep = dep / self.args.max_depth
                pred = pred / self.args.max_depth
                pred_gray = pred
                gt = gt / self.args.max_depth

                rgb = np.clip(rgb, 0, 256).astype('uint8')
                dep = (255.0*cm(dep)).astype('uint8')
                pred = (255.0*cm(pred)).astype('uint8')
                pred_gray = (255.0*pred_gray).astype('uint8')
                gt = (255.0*cm(gt)).astype('uint8')

                rgb = Image.fromarray(rgb, 'RGB')
                dep = Image.fromarray(dep[:, :, :3], 'RGB')
                pred = Image.fromarray(pred[:, :, :3], 'RGB')
                pred_gray = Image.fromarray(pred_gray)
                gt = Image.fromarray(gt[:, :, :3], 'RGB')


                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)

                path_save_rgb = '{}/01_rgb.png'.format(self.path_output)
                path_save_dep = '{}/02_dep.png'.format(self.path_output)
                path_save_pred = '{}/05_pred_final.png'.format(self.path_output)
                path_save_pred_gray = '{}/05_pred_final_gray.png'.format(
                    self.path_output)
                path_save_gt = '{}/06_gt.png'.format(self.path_output)

                rgb.save(path_save_rgb)
                dep.save(path_save_dep)
                pred.save(path_save_pred)
                pred_gray.save(path_save_pred_gray)
                gt.save(path_save_gt)
