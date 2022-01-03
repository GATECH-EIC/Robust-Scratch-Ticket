import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args


DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.use_subset = True

        self.score_mask = None

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def init_weight_with_score(self, prune_rate):
        self.weight.data = self.weight.data * GetSubnet.apply(self.clamped_scores, prune_rate).data
        self.use_subset = False

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return GetSubnet.apply(self.clamped_scores, self.prune_rate).detach()

    def discard_low_score(self, discard_rate):
        self.score_mask = GetSubnet.apply(self.clamped_scores, 1-discard_rate).detach() == 0
        self.scores[self.score_mask].data.zero_()
        self.scores.grad[self.score_mask] = 0

    def clear_low_score_grad(self):
        if self.score_mask is not None:
            self.scores.grad[self.score_mask] = 0

    def clear_subset_grad(self):
        subset = self.get_subnet()
        mask = subset == 1
        self.weight.grad[mask] = 0

    def lr_scale_zero(self, lr_scale):
        subset = self.get_subnet()
        mask = subset == 0
        self.weight.grad[mask].data *= lr_scale

    def weight_decay_custom(self, weight_decay, weight_decay_on_zero):
        subset = self.get_subnet()
        mask = subset == 1

        l2_reg_subset = torch.norm(self.weight[mask])
        l2_reg_zero = torch.norm(self.weight[~mask])

        loss = weight_decay * l2_reg_subset + weight_decay_on_zero * l2_reg_zero
        loss.backward()

    def forward(self, x):
        if self.use_subset:
            subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * subnet
        else:
            w = self.weight

        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x



class SubnetConv_filter(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[:1]))
        nn.init.normal_(self.scores, mean=0.0, std=1.0)

        self.use_subset = True

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def init_weight_with_score(self, prune_rate):
        self.weight.data = self.weight.data * GetSubnet.apply(self.clamped_scores, prune_rate).view(self.weight.size()[0], 1, 1, 1).data
        self.use_subset = False

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def get_subset(self):
        return GetSubnet.apply(self.clamped_scores, self.prune_rate).detach()

    def clear_subset_grad(self):
        subset = self.get_subnet()
        mask = subset == 1
        self.weight.grad[mask] = 0

    def lr_scale_zero(self, lr_scale):
        subset = self.get_subnet()
        mask = subset == 0
        self.weight.grad[mask].data *= lr_scale

    def weight_decay_custom(self, weight_decay, weight_decay_on_zero):
        subset = self.get_subnet()
        mask = subset == 1

        l2_reg_subset = torch.norm(self.weight[mask])
        l2_reg_zero = torch.norm(self.weight[1-mask])

        loss = weight_decay * l2_reg_subset + weight_decay_on_zero * l2_reg_zero
        loss.backward()

    def forward(self, x):
        if self.use_subset:
            subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * subnet.view(self.weight.size()[0], 1, 1, 1)
        else:
            w = self.weight

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetConv_kernel(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[:2]))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.use_subset = True

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def init_weight_with_score(self, prune_rate):
        self.weight.data = self.weight.data * GetSubnet.apply(self.clamped_scores, prune_rate).view(self.weight.size()[0], self.weight.size()[1], 1, 1).data
        self.use_subset = False
        
    @property
    def clamped_scores(self):
        return self.scores.abs()

    def get_subset(self):
        return GetSubnet.apply(self.clamped_scores, self.prune_rate).detach()

    def clear_subset_grad(self):
        subset = self.get_subnet()
        mask = subset == 1
        self.weight.grad[mask] = 0

    def lr_scale_zero(self, lr_scale):
        subset = self.get_subnet()
        mask = subset == 0
        self.weight.grad[mask].data *= lr_scale

    def weight_decay_custom(self, weight_decay, weight_decay_on_zero):
        subset = self.get_subnet()
        mask = subset == 1

        l2_reg_subset = torch.norm(self.weight[mask])
        l2_reg_zero = torch.norm(self.weight[1-mask])

        loss = weight_decay * l2_reg_subset + weight_decay_on_zero * l2_reg_zero
        loss.backward()

    def forward(self, x):
        if self.use_subset:
            subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * subnet.view(self.weight.size()[0], self.weight.size()[1], 1, 1)
        else:
            w = self.weight

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SubnetConv_row(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[:3]))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.use_subset = True
        
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def init_weight_with_score(self, prune_rate):
        self.weight.data = self.weight.data * GetSubnet.apply(self.clamped_scores, prune_rate).view(self.weight.size()[0], self.weight.size()[1], self.weight.size()[2], 1).data
        self.use_subset = False

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def get_subset(self):
        return GetSubnet.apply(self.clamped_scores, self.prune_rate).detach()

    def clear_subset_grad(self):
        subset = self.get_subnet()
        mask = subset == 1
        self.weight.grad[mask] = 0

    def lr_scale_zero(self, lr_scale):
        subset = self.get_subnet()
        mask = subset == 0
        self.weight.grad[mask].data *= lr_scale
        
    def weight_decay_custom(self, weight_decay, weight_decay_on_zero):
        subset = self.get_subnet()
        mask = subset == 1

        l2_reg_subset = torch.norm(self.weight[mask])
        l2_reg_zero = torch.norm(self.weight[1-mask])

        loss = weight_decay * l2_reg_subset + weight_decay_on_zero * l2_reg_zero
        loss.backward()

    def forward(self, x):
        if self.use_subset:
            subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
            w = self.weight * subnet.view(self.weight.size()[0], self.weight.size()[1], self.weight.size()[2], 1)
        else:
            w = self.weight

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        subnet, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs[subnet == 0.0] = 0.0

        return grad_inputs, None


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

