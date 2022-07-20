from . import BaseLoss
import torch
from .uncertrainty import Ucertl2MaskedMSELoss
from torch import nn


class Loss(BaseLoss):
    def __init__(self, args):
        super(Loss, self).__init__(args)

        self.loss_name = []

        for k, _ in self.loss_dict.items():
            self.loss_name.append(k)

    def compute(self, sample, output):
        loss_val = []

        for loss_type in self.loss_dict:
            loss = self.loss_dict[loss_type]
            loss_func = loss['func']
            if loss_func is None:
                continue

            gt = sample['gt']

            loss_tmp = loss['weight'] * loss_func(output, gt)
            loss_val.append(loss_tmp)

        loss_val = torch.stack(loss_val)
        loss_sum = torch.sum(loss_val, dim=0, keepdim=True)

        # Accumulate loss
        loss_val = torch.cat((loss_val, loss_sum))
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()

        return loss_sum, loss_val


class MyLoss(BaseLoss):
    def __init__(self, args):
        super(MyLoss, self).__init__(args)
        self.loss_name = ['Uncertainty lOSS']
        self.loss = Ucertl2MaskedMSELoss()
        self.pool = nn.MaxPool2d((2, 2))

    def compute(self, sample, output):
        pred_1, pred_2, pred_3, pred_4, s_1, s_2, s_3, s_4 = output
        epoch = sample['epoch']
        gt4 = gt = sample['gt']
        gt3 = self.pool(gt4)
        gt2 = self.pool(gt3)
        gt1 = self.pool(gt2)
        loss_val = []

        if epoch < 30:
            depth_loss = 0.40 * self.loss(pred_4, s_4, gt4) \
                         + 0.30 * self.loss(pred_3, s_3, gt3) \
                         + 0.20 * self.loss(pred_2, s_2, gt2)  \
                         + 0.10 * self.loss(pred_1, s_1, gt1)
        elif epoch < 60:
            depth_loss = 0.55 * self.loss(pred_4, s_4, gt4) \
                         + 0.25 * self.loss(pred_3, s_3, gt3) \
                         + 0.15 * self.loss(pred_2, s_2, gt2) \
                         + 0.05 * self.loss(pred_1, s_1, gt1)
        else:
            depth_loss = self.loss(pred_4, s_4, gt4)
        loss_sum = depth_loss
        loss_val.append(depth_loss)
        loss_val = torch.stack(loss_val)
        # Accumulate loss
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()
        return loss_sum, loss_val


class UcertRELossL1(nn.Module):
    def __init__(self):
        super(UcertRELossL1, self).__init__()
        self.loss_name = ['UcertRELossL1']

    def forward(self, pred, predm, Ucert, target):
        loss_val = []
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred - predm
        diff = diff[valid_mask]
        Ucert = Ucert[valid_mask] + 1.
        self.loss = (diff * Ucert).abs().mean()

        loss_sum = self.loss
        loss_val.append(loss_sum)
        loss_val = torch.stack(loss_val)
        # Accumulate loss
        loss_val = torch.unsqueeze(loss_val, dim=0).detach()
        return loss_sum, loss_val