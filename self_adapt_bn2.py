import torch.nn as nn
from copy import deepcopy

import functools
import torch
from collections import deque
import torch.nn.functional as F
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def freeze_layers(opts, model: torch.nn.Module):
    if len(opts.resnet_layers) != 0 and "resnet" in opts.backbone_name and "deeplab" in opts.arch_type:
        if opts.arch_type == "deeplab":
            model_name = model.module
        elif opts.arch_type == "deeplabv3plus":
            model_name = model.module.model
        else:
            raise ValueError(f"{opts.arch_type} not compatible with resnet layer freezing")
        for idx, p in enumerate(model_name.named_parameters()):
            if idx <= 3:
                p[1].requires_grad = False
            else:
                break
        for layer in opts.resnet_layers:
            layer = "layer" + str(layer)
            for para in getattr(model_name.backbone, layer).named_parameters():
                para[1].requires_grad = False
    if len(opts.hrnet_layers) != 0 and "hrnet" in opts.arch_type:
        for idx, p in enumerate(model.module.model.named_parameters()):
            if idx <= 3:
                p[1].requires_grad = False
            else:
                break
        for layer_idx in opts.hrnet_layers:
            layer = "transition" + str(layer_idx)
            for para in getattr(model.module.model, layer).named_parameters():
                para[1].requires_grad = False
            layer = "stage" + str(layer_idx + 1)
            for para in getattr(model.module.model, layer).named_parameters():
                para[1].requires_grad = False



# class SelfAdaptiveNormalization(nn.Module):
#     def __init__(self,
#                  num_features: int,
#                  index: int,
#                  unweighted_stats: bool = False,
#                  eps: float = 1e-5,
#                  momentum: float = 0.1,
#                  alpha: float = 0.5,
#                  alpha_train: bool = True,
#                  affine: bool = True,
#                  track_running_stats: bool = True,
#                  training: bool = True,
#                  update_source: bool = True,
#                  k: int = 12):
#         super(SelfAdaptiveNormalization, self).__init__()
#         self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=alpha_train)
#         self.alpha_train = alpha_train
#         self.training = training
#         self.unweighted_stats = unweighted_stats
#         self.eps = eps
#         self.track_running_stats = track_running_stats
#         self.update_source = update_source
#         self.batch_norm = nn.BatchNorm2d(
#             num_features,
#             eps,
#             momentum,
#             affine,
#             track_running_stats
#         )
#         self.k = k
#         self.index = index
#         self.register_buffer('all_running_mean', torch.zeros(k, num_features))
#         self.register_buffer('all_running_var', torch.zeros(k, num_features))
        
#     def update_task_vector(self, task_index):
#         if task_index < self.k:
#             self.all_running_mean[task_index] = self.batch_norm.running_mean.clone().detach()
#             self.all_running_var[task_index] = self.batch_norm.running_var.clone().detach()
#         else:
#             raise IndexError('Index out of range!')

#     def get_stats_queue(self):
#         return list(self.mean_queue), list(self.var_queue)
    
#     def get_weight_mean_var(self, x_mean, x_var):
#         # print(self.all_running_mean[:self.index], self.all_running_mean[self.index:])
#         mean_queue = torch.mean(torch.cat([self.all_running_mean[:self.index], x_mean.unsqueeze(0)], dim=0), dim=0)
#         var_queue = torch.mean(torch.cat([self.all_running_var[:self.index], x_var.unsqueeze(0)], dim=0), dim=0)
#         return mean_queue, var_queue

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if (not self.training and not self.unweighted_stats) or (self.training and self.alpha_train):
#             # if self.alpha_train:
#             #     self.alpha.requires_grad_()
#             self.alpha.requires_grad_(False)
#             # Compute statistics from batch
#             x_mean = torch.mean(x, dim=(0, 2, 3))
#             x_var = torch.var(x, dim=(0, 2, 3), unbiased=False)
            
#             weighted_mean, weighted_var = self.get_weight_mean_var(x_mean, x_var)

#             # Weigh batch statistics with running statistics
#             alpha = torch.clamp(self.alpha, 0, 1)
#             weighted_mean = (1 - alpha) * weighted_mean + alpha * x_mean
#             weighted_var = (1 - alpha) * weighted_var + alpha * x_var
#             if self.alpha == 0.:
#                 weighted_mean = self.batch_norm.running_mean
#                 weighted_var = self.batch_norm.running_var
#             # Update running statistics based on momentum
#             if self.update_source and self.training:
#                 self.batch_norm.running_mean = (1 - self.batch_norm.momentum) * self.batch_norm.running_mean\
#                                                + self.batch_norm.momentum * x_mean
#                 self.batch_norm.running_var = (1 - self.batch_norm.momentum) * self.batch_norm.running_var\
#                                               + self.batch_norm.momentum * x_var
#             return compute_bn(
#                 x, weighted_mean, weighted_var,
#                 self.batch_norm.weight, self.batch_norm.bias, self.eps
#                 )
#         x = self.batch_norm(x)
#         return x

# def compute_bn(input: torch.Tensor, weighted_mean: torch.Tensor, weighted_var: torch.Tensor,
#                weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
#     input = (input - weighted_mean[None, :, None, None]) / (torch.sqrt(weighted_var[None, :, None, None] + eps))
#     input = input * weight[None, :, None, None] + bias[None, :, None, None]
#     return input


# def replace_batchnorm(m: torch.nn.Module,
#                       alpha: float,
#                       index: int,
#                       update_source_bn: bool = True):
#     if alpha is None:
#         alpha = 0.0
#     for name, child in m.named_children():
#         if isinstance(child, torch.nn.BatchNorm2d):
#             wbn = SelfAdaptiveNormalization(num_features=child.num_features,
#                                      alpha=alpha, index=index, update_source=update_source_bn)

#             setattr(wbn.batch_norm, "running_mean", deepcopy(child.running_mean))
#             setattr(wbn.batch_norm, "running_var", deepcopy(child.running_var))
#             setattr(wbn.batch_norm, "weight", deepcopy(child.weight))
#             setattr(wbn.batch_norm, "bias", deepcopy(child.bias))

#             wbn.to(next(m.parameters()).device.type)
#             setattr(m, name, wbn)
#             # if child.num_features>190 or alpha!=0:
#             #     wbn = RN_L(feature_channels=child.num_features, originBN=child)

#             #     wbn.to(next(m.parameters()).device.type)
#             #     setattr(m, name, wbn)
            
#         else:
#             replace_batchnorm(child, alpha=alpha, index=index, update_source_bn=update_source_bn)




class SelfAdaptiveNormalization(nn.Module):
    def __init__(self,
                 originBN: int,
                 index: int,
                 unweighted_stats: bool = False,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 alpha: float = 0.3,
                 k: int = 12,
                 alpha_train: bool = True,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 training: bool = True,
                 update_source: bool = True):
        super(SelfAdaptiveNormalization, self).__init__()
        self.alpha = torch.tensor(alpha)#nn.Parameter(torch.tensor(alpha), requires_grad=alpha_train)
        self.alpha_train = alpha_train
        self.training = training
        self.unweighted_stats = unweighted_stats
        self.eps = eps
        self.update_source = update_source
        # self.batch_norm  = originBN
        self.batch_norm = nn.BatchNorm2d(num_features=originBN.num_features, eps=originBN.eps, momentum=originBN.momentum, affine=True, track_running_stats=False)
        self.batch_norm.weight = deepcopy(originBN.weight)
        self.batch_norm.bias = deepcopy(originBN.bias)
        self.batch_norm.requires_grad_(True)
        self.flag = True
        self.k = k
        num_features = self.batch_norm.num_features
        self.index = index
        # self.register_buffer('all_running_mean', torch.zeros(k, num_features))
        # self.register_buffer('all_running_var', torch.ones(k, num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.running_mean = deepcopy(originBN.running_mean)
        self.running_var = deepcopy(originBN.running_var)
        self.track_running_stats = True
        self.num_batches_tracked = originBN.num_batches_tracked
        self.momentum = originBN.momentum
        self.origin = originBN.requires_grad_(False)
        self.normed_div_mean = self.alpha
        
        
    def update_task_vector(self, task_index, task_mean, task_var, momentum=None):
        # if momentum is None:
        #     momentum = self.batch_norm.momentum
        if task_index < self.k:
            if torch.all(self.all_running_mean[task_index] == 0):
                self.all_running_mean[task_index] = task_mean.clone().detach()
                self.all_running_var[task_index] = task_var.clone().detach()
            else:
                self.all_running_mean[task_index] = (1 - momentum) * self.all_running_mean[task_index]\
                                                + momentum * task_mean.clone().detach()
                self.all_running_var[task_index] = (1 - momentum) * self.all_running_var[task_index]\
                                                + momentum * task_var.clone().detach()
            # self.all_running_mean[task_index] = task_mean.clone().detach()
            # self.all_running_var[task_index] = task_var.clone().detach()
        else:
            raise IndexError('Index out of range!')

    def get_weight_mean_var(self, x_mean, x_var):
        # print(self.all_running_mean[:self.index], self.all_running_mean[self.index:])
        # print(self.index)
        # last_vector = self.all_running_mean[self.index].unsqueeze(0) 
        # other_vectors = self.all_running_mean[:self.index+1]
        # similarities = F.cosine_similarity(last_vector, other_vectors, dim=1)
        # epsilon = 1e-6
        # weights = 1 / (similarities + epsilon)
        # weights /= weights.sum()

        # # print(index)
        # expanded_weights = weights.unsqueeze(1).expand_as(other_vectors)
        # mean_queue = (expanded_weights * other_vectors).sum(dim=0)
        
        # last_vector = self.all_running_var[self.index].unsqueeze(0) 
        # other_vectors = self.all_running_var[:self.index+1]
        # similarities = F.cosine_similarity(last_vector, other_vectors, dim=1)
        # epsilon = 1e-6
        # weights = 1 / (similarities + epsilon)
        # weights /= weights.sum()

        # # print(index)
        # expanded_weights = weights.unsqueeze(1).expand_as(other_vectors)
        # var_queue = (expanded_weights * other_vectors).sum(dim=0)
        prior = [1,4,6,8]

        filtered_list = [x for x in prior if x < self.index]
        mean_queue = self.all_running_mean[filtered_list].mean(0)
        var_queue = self.all_running_var[filtered_list].mean(0)

        return mean_queue, var_queue
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.batch_norm.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use the true average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use a custom momentum
                    exponential_average_factor = self.momentum
        elif self.batch_norm.training or not self.track_running_stats:
            exponential_average_factor = 1.0

        if self.batch_norm.training:
            bs = x.shape[0]//2
            x_mean = x.mean((0, 2, 3), keepdims=False)
            x_var = x.var((0, 2, 3), keepdims=False, unbiased=True)

            if self.flag:
                # weighted_mean, weighted_var = self.get_weight_mean_var(x_mean, x_var)

                # Weigh batch statistics with running statistics
                # alpha = torch.clamp(self.alpha, 0, 1)
                # prior = [1,4,6]
                # if self.index in prior:
                #     self.alpha = torch.tensor(0.1)
                with torch.no_grad():
                    cur_mean = x[:bs].mean((0, 2, 3), keepdims=False)
                    cur_var = x[:bs].var((0, 2, 3), keepdims=False, unbiased=True)
                    pre_mean = x[bs:].mean((0, 2, 3), keepdims=False)
                    pre_var = x[bs:].var((0, 2, 3), keepdims=False, unbiased=True)
                    source_distribution = torch.distributions.MultivariateNormal(self.origin.running_mean, (
                            self.origin.running_var + 0.00001) * torch.eye(
                        self.origin.running_var.shape[0]).cuda())
                    pre_distribution = torch.distributions.MultivariateNormal(pre_mean, (
                                pre_var + 0.00001) * torch.eye(
                            pre_var.shape[0]).cuda())
                    cur_distribution = torch.distributions.MultivariateNormal(cur_mean, (
                                cur_var + 0.00001) * torch.eye(
                            cur_var.shape[0]).cuda())
                    target_distribution = torch.distributions.MultivariateNormal(x_mean, (
                                x_var + 0.00001) * torch.eye(
                            x_var.shape[0]).cuda())
                    div1 = (0.5 * torch.distributions.kl_divergence(source_distribution,pre_distribution) + 0.5 * torch.distributions.kl_divergence(pre_distribution, source_distribution))
                    div2 = (0.5 * torch.distributions.kl_divergence(source_distribution,cur_distribution) + 0.5 * torch.distributions.kl_divergence(cur_distribution, source_distribution))
                    div3 = (0.5 * torch.distributions.kl_divergence(source_distribution,target_distribution) + 0.5 * torch.distributions.kl_divergence(target_distribution, source_distribution))
                    
                    self.normed_div_mean=div3/(div1+div2)
                    # print(self.normed_div_mean)
                    self.normed_div_mean = torch.clamp(self.normed_div_mean, 0, 1)
                
                # weighted_mean = (1 - alpha) * self.running_mean + alpha * x_mean
                # weighted_var = (1 - alpha) * self.running_var + alpha * x_var
                # self.update_task_vector(self.index, x[:bs].mean((0, 2, 3), keepdims=False), x[:bs].var((0, 2, 3), keepdims=False, unbiased=True), momentum=exponential_average_factor)
            weighted_mean = (1 - self.normed_div_mean) * self.origin.running_mean + self.normed_div_mean * x_mean
            weighted_var = (1 - self.normed_div_mean) * self.origin.running_var + self.normed_div_mean * x_var

            with torch.no_grad():
                self.running_mean = (1 - exponential_average_factor) * self.running_mean\
                                               + exponential_average_factor * weighted_mean
                self.running_var = (1 - exponential_average_factor) * self.running_var\
                                              + exponential_average_factor * weighted_var
                # self.update_task_vector(self.index, self.running_mean, self.running_var, exponential_average_factor)

            return compute_bn(
                x,
                weighted_mean,
                weighted_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                self.batch_norm.eps,
            )
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                False,
                0.0,
                self.batch_norm.eps,
            )
class CustomBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(CustomBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use the true average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use a custom momentum
                    exponential_average_factor = self.momentum
        elif self.training or not self.track_running_stats:
            exponential_average_factor = 1.0

        # Calculate the mean and variance of the current batch
        mean = input.mean([0, 2, 3])
        var = input.var([0, 2, 3], unbiased=False)

        # Update the running mean and variance using a custom method
        with torch.no_grad():
            self.running_mean = exponential_average_factor * self.running_mean + (1 - exponential_average_factor) * mean
            self.running_var = exponential_average_factor * self.running_var + (1 - exponential_average_factor) * var

        # Normalization
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input
def compute_bn(input: torch.Tensor, weighted_mean: torch.Tensor, weighted_var: torch.Tensor,
               weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    input = (input - weighted_mean[None, :, None, None]) / (torch.sqrt(weighted_var[None, :, None, None] + eps))
    input = input * weight[None, :, None, None] + bias[None, :, None, None]
    return input


def replace_batchnorm(m: torch.nn.Module,
                      index: float,
                    #   alpha: int,
                      update_source_bn: bool = True):
    for name, child in m.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            wbn = SelfAdaptiveNormalization(child, index)

            # setattr(wbn, "running_mean", deepcopy(child.running_mean))
            # setattr(wbn, "running_var", deepcopy(child.running_var))
            # setattr(wbn, "weight", deepcopy(child.weight))
            # setattr(wbn, "bias", deepcopy(child.bias))
            wbn.to(next(m.parameters()).device.type)
            setattr(m, name, wbn)
        else:
            replace_batchnorm(child, index, update_source_bn=update_source_bn)

def reinit_alpha(m: torch.nn.Module,
                 alpha: float,
                 device: torch.device,
                 alpha_train: bool = False):

    layers = [module for module in m.modules() if isinstance(module, SelfAdaptiveNormalization)]
    for i, layer in enumerate(layers):
        layer.alpha = nn.Parameter(torch.tensor(alpha).to(device), requires_grad=alpha_train)


def set_batchnorm(m: torch.nn.Module,
                      flag: bool):

    for name, child in m.named_children():
        if isinstance(child, SelfAdaptiveNormalization):
            child.flag = flag
        else:
            set_batchnorm(child, flag=flag)
def set_eval(m: torch.nn.Module,
                      flag: bool):

    for name, child in m.named_children():
        if isinstance(child, SelfAdaptiveNormalization):
            child.training = flag
        else:
            set_eval(child, flag=flag)
def update_batchnorm(m: torch.nn.Module,
                      index: int):

    for name, child in m.named_children():
        if isinstance(child, SelfAdaptiveNormalization):
            child.update_task_vector(index, child.origin.running_mean, child.origin.running_var)
        else:
            update_batchnorm(child, index=index)
