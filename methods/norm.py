import torch
import torch.nn as nn
from copy import deepcopy
from methods.base import TTAMethod
from methods.bn import AlphaBatchNorm, EMABatchNorm,BayesianBatchNorm
from conf import cfg
import math
import csv
def append_tensor_to_csv(tensor_value, file_path='data.csv'):
    # Convert the tensor to a list of lists (a single row)
    tensor_value_list = [tensor_value.cpu().detach().tolist()]

    # Append the tensor value to the CSV file
    with open(file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(tensor_value_list)

class Norm(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.EMA_normed_div_mean = torch.zeros(int(sum(1 for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))/2)).cuda()
        self.index = 0
    @torch.no_grad()
    def forward_and_adapt(self, x):
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, BayesianBatchNorm):
                    m.norm.train()
            imgs_test = x[0]
            self.m = 1/2

            div_mean = []
            _ = self.model(imgs_test)

            for name, module in self.model.named_modules():
                if isinstance(module, BayesianBatchNorm):
                    div_mean.append(module.div_values)

            normed_div_mean = scale_to_mean_std(div_mean)
            tructed_div = [max(min(x, torch.tensor(1).cuda()), torch.tensor(-1).cuda()) for x in normed_div_mean]

            ii = 0
            for name, module in self.model.named_modules():
                if isinstance(module, BayesianBatchNorm):
                    module.normed_div_mean = (tructed_div[ii]+1)/2 * self.m
                    ii += 1
            prediction = self.model(imgs_test)

            self.EMA_normed_div_mean = self.EMA_normed_div_mean * 0.9 + ( (torch.stack(tructed_div) + 1) / 2 * self.m) * 0.1
            jj = 0
            for name, module in self.model.named_modules():
                if isinstance(module, BayesianBatchNorm):
                    module.normed_div_mean = self.EMA_normed_div_mean[jj]
                    jj += 1

        return prediction

    def copy_model_and_optimizer(self):
        """Copy the model and optimizer states for resetting after adaptation."""
        self.model_state = deepcopy(self.model.state_dict())
        return self.model_state, None

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        if self.cfg.MODEL.ADAPTATION == "norm_test":  # BN--1
            for m in self.model.modules():
                # Re-activate batchnorm layer
                if (isinstance(m, nn.BatchNorm1d) and self.batch_size > 1) or isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train()
        elif self.cfg.MODEL.ADAPTATION == "norm_alpha":  # BN--0.1
            # (1-alpha) * src_stats + alpha * test_stats
            self.model = AlphaBatchNorm.adapt_model(self.model, alpha=self.cfg.BN.ALPHA).cuda()
        elif self.cfg.MODEL.ADAPTATION == "norm_ema":  # BN--EMA
            self.model = EMABatchNorm.adapt_model(self.model).cuda()
        elif self.cfg.MODEL.ADAPTATION == "TTBN":  # BN--ours
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = 0.0
            self.model = BayesianBatchNorm.adapt_model(self.model,prior=1).cuda()


def scale_to_mean_std(numbers):
    mean = sum(numbers) / len(numbers)
    std = math.sqrt(sum((x - mean) ** 2 for x in numbers) / len(numbers))
    scaled_numbers = [(x - mean) / std for x in numbers]
    return scaled_numbers