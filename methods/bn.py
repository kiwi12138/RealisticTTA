# Copyright 2020-2021 Evgenia Rusak, Steffen Schneider, George Pachitariu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ---
# This licence notice applies to all originally written code by the
# authors. Code taken from other open-source projects is indicated.
# See NOTICE for a list of all third-party licences used in the project.

"""Batch norm variants
AlphaBatchNorm builds upon: https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py
"""

from torch import nn
from torch.nn import functional as F
import torch
from conf import cfg

class AlphaBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """

    @staticmethod
    def find_bns(parent, alpha):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = AlphaBatchNorm(child, alpha)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(AlphaBatchNorm.find_bns(child, alpha))

        return replace_mods

    @staticmethod
    def adapt_model(model, alpha):
        replace_mods = AlphaBatchNorm.find_bns(model, alpha)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, alpha):
        assert alpha >= 0 and alpha <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.alpha = alpha

        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False, momentum=1.0)

    def forward(self, input):
        self.norm(input)

        running_mean = ((1 - self.alpha) * self.layer.running_mean + self.alpha * self.norm.running_mean)
        running_var = ((1 - self.alpha) * self.layer.running_var + self.alpha * self.norm.running_var)

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )


class EMABatchNorm(nn.Module):
    @staticmethod
    def adapt_model(model):
        model = EMABatchNorm(model)
        return model

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # store statistics, but discard result
        self.model.train()
        self.model(x)
        # store statistics, use the stored stats
        self.model.eval()
        return self.model(x)

class BayesianBatchNorm(nn.Module):
    """ Use the source statistics as a prior on the target statistics """
    @staticmethod
    def find_bns(parent, prior):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            if isinstance(child, nn.BatchNorm2d):
                module = BayesianBatchNorm(child, prior)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(BayesianBatchNorm.find_bns(child, prior))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior):
        replace_mods = BayesianBatchNorm.find_bns(model, prior)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1
        super(BayesianBatchNorm, self).__init__()
        self.layer = layer
        self.layer.eval()
        self.norm = nn.BatchNorm2d(self.layer.num_features, affine=False,momentum=cfg.TEST.MOMENTUM,track_running_stats=True).cuda() #todo
        self.normed_div_mean = torch.zeros(1).cuda()

    def forward(self, input):
        # if self.norm.training is True:
        self.norm(input)
        self.norm.eval()
        source_distribution = torch.distributions.MultivariateNormal(self.layer.running_mean, (
                self.layer.running_var + 0.00001) * torch.eye(
            self.layer.running_var.shape[0]).cuda())
        target_distribution = torch.distributions.MultivariateNormal(self.norm.running_mean, (
                self.norm.running_var + 0.00001) * torch.eye(
            self.norm.running_var.shape[0]).cuda())

        self.div = (0.5 * torch.distributions.kl_divergence(source_distribution,target_distribution) + 0.5 * torch.distributions.kl_divergence(target_distribution, source_distribution))

        self.div_values = self.div
        self.prior = self.normed_div_mean

        running_mean = (self.prior * self.layer.running_mean+ (1 - self.prior) * self.norm.running_mean)
        running_var = (self.prior * self.layer.running_var) + (1 - self.prior) * self.norm.running_var + self.prior * (1 - self.prior) * ((self.layer.running_mean - self.norm.running_mean) ** (2))

        output = (input - running_mean[None, :, None, None]) / torch.sqrt(
            running_var[None, :, None, None] + self.layer.eps) * self.layer.weight[None, :, None, None] + self.layer.bias[None, :, None, None]

        return output
