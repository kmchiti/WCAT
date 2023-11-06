import torch
import torch.nn as nn
import torch.nn.functional as F
from common import FromParams, Registrable, Params
import numpy as np
from scipy.optimize import minimize
import json


def int2bin(input: torch.tensor, num_bits: int):
    """
    code from: https://github.com/elliothe/BFA
    convert the signed integer value into unsigned integer (2's complement equivalently).
    Note that, the conversion is different depends on number of bit used.
    """
    output = input.clone()
    if num_bits == 1:  # when it is binary, the conversion is different
        output = output / 2 + .5
    elif num_bits > 1:
        output[input.lt(0)] = 2 ** num_bits + output[input.lt(0)]

    return output


def bin2int(input: torch.tensor, num_bits: int):
    """
    code from: https://github.com/elliothe/BFA
    convert the unsigned integer (2's complement equivalently) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    """
    if num_bits == 1:
        output = input * 2 - 1
    elif num_bits > 1:
        mask = 2 ** (num_bits - 1) - 1
        output = -(input & ~mask) + (input & mask)
    else:
        raise "N_bits should be >0"
    return output


def binary(input: torch.tensor, num_bits: int):
    """
    Args:
        input: an unsigned tensor in shape of [d]
        num_bits: bit precision

    Returns:
        2D tensor in shape of [d * b]
    example:
    >>x = torch.tensor([1,3,0,2])
    >>binary(x, bits=2)
    tensor([[0, 1],
        [1, 1],
        [0, 0],
        [1, 0]], dtype=torch.uint8)

    """
    if torch.is_floating_point(input):
        assert "input data type should be integer."
    mask = 2 ** torch.arange(num_bits - 1, -1, -1).to(input.device, input.dtype)
    return input.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def hamming_distance(x_1: torch.tensor, x_2: torch.tensor, num_bits: int):
    """
    Args:
        x_1: first integer tensor
        x_2: second integer tensor
        num_bits: bit precision

    Returns:
        H_dist: hamming distance between two int tensors
    """
    assert x_1.dtype is torch.int16
    assert x_2.dtype is torch.int16
    H_dist = binary(x_1 ^ x_2, num_bits).sum().item()
    return H_dist


def get_scale(input: torch.tensor, N_bits: int = 2):
    """
    extract optimal scale based on statistics of the input tensor.
    from: https://arxiv.org/pdf/1805.06085.pdf
    Args:
        input: input real tensor
        N_bits: bit precision
    Returns:
        scale: optimal scale
    """
    assert N_bits in [2, 4, 8]
    z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit': [0.027, 1.114]}
    z = z_typical[f'{N_bits}bit']
    c1, c2 = 1 / z[0], z[1] / z[0]
    std = input.std()
    mean = input.abs().mean()
    q_scale = c1 * std - c2 * mean
    return q_scale.detach()


def sample_noise(noise_level: float, shape: tuple, device: torch.device, noise_model: str = "bernoulli"):
    """
    Args:
        noise_level: float number indication bit error rate
        shape: list of dimensions [... * N_bits]
        device:
        noise_model:

    Returns:
        epsilon: integer tensor

    """
    bw = 2 ** torch.arange(shape[-1]).short()
    if noise_model == "bernoulli":
        epsilon = torch.bernoulli(torch.ones(size=shape) * noise_level).short()
    elif noise_model == "uniform":
        uniform = torch.distributions.Uniform(torch.tensor([0.]), torch.tensor([1.]))
        unif_sample = uniform.sample(shape)[:, :, 0]
        epsilon = torch.zeros_like(unif_sample).to(torch.int)
        epsilon[unif_sample < noise_level] = 1
    else:
        raise "didn't find noise model"
    epsilon = epsilon @ bw
    return epsilon.to(device)


def round_pass(input: torch.tensor):
    """
    Args:
        input: input tensor

    Returns:
        rounded tensor with STE for backward
    """
    y = input.round()
    y_grad = input
    return (y - y_grad).detach() + y_grad


def grad_scale(input: torch.tensor, scale: float):
    """
        Args:
            input: input tensor
            scale: gradient scale for backward

        Returns:
            rounded tensor with STE for backward
        """
    y = input
    y_grad = input * scale
    return (y - y_grad).detach() + y_grad


class _Clamp(torch.autograd.Function, Registrable):

    @staticmethod
    def forward(ctx, input, q_range, signed=True):
        """
        Args:
            ctx: a context object that can be used to stash information for backward computation
            input: input tensor
            signed: flag to indicate signed ot unsigned quantization
            q_range: scale factor

        Returns:
            clipped tensor
        """
        ctx.q_range = q_range
        ctx.input = input.clone()
        if signed:
            return input.clamp(-q_range, q_range)
        else:
            return input.clamp(torch.tensor(0.).to(q_range.device), q_range)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    def __reduce__(self):
        # This method is called during pickling to serialize the object
        return (_Clamp, ())


@_Clamp.register("ste")
class _Clamp_STE(_Clamp):

    @staticmethod
    def backward(ctx, grad_output):
        q_range_grad = -1. * (ctx.input < -ctx.q_range) + 1. * (ctx.input > ctx.q_range)
        input_grad = 1.
        return input_grad * grad_output, q_range_grad * grad_output, None

    def __reduce__(self):
        # This method is called during pickling to serialize the object
        return (_Clamp_STE, ())


@_Clamp.register("pwl")
class _Clamp_PWL(_Clamp):

    @staticmethod
    def backward(ctx, grad_output):
        q_range_grad = -1 * (ctx.input < -ctx.q_range) + 1 * (ctx.input > ctx.q_range)
        input_grad = 1. * (ctx.input.abs() <= ctx.q_range) + 0. * (ctx.input.abs() > ctx.q_range)
        return input_grad * grad_output, q_range_grad * grad_output, None

    def __reduce__(self):
        # This method is called during pickling to serialize the object
        return (_Clamp_PWL, ())


@_Clamp.register("mad")
class _Clamp_MAD(_Clamp):

    @staticmethod
    def backward(ctx, grad_output):
        q_range_grad = -1 * (ctx.input < -ctx.q_range) + 1 * (ctx.input > ctx.q_range)
        input_grad = 1. * (ctx.input.abs() <= ctx.q_range) + ctx.q_range / ctx.input.abs() * (
                ctx.input.abs() > ctx.q_range)
        return input_grad * grad_output, q_range_grad * grad_output, None

    def __reduce__(self):
        # This method is called during pickling to serialize the object
        return (_Clamp_MAD, ())


class Quantizer(torch.nn.Module, Registrable):
    def __init__(self, N_bits: int = 4, signed: bool = True, p0: float = 0.):
        super().__init__()
        self.N_bits = N_bits
        self.signed = signed
        self.p0 = p0
        self.max_range = 0.
        self.min_range = 0.
        if self.signed:
            self.Qn = - 2 ** (self.N_bits - 1)
            self.Qp = 2 ** (self.N_bits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2 ** self.N_bits - 1

    def linear_quantize(self, input: torch.tensor):
        raise NotImplementedError

    def linear_dequantize(self, input: torch.tensor):
        raise NotImplementedError

    def _init_q_params(self, input: torch.tensor):
        raise NotImplementedError

    def monitor_ranges(self):
        raise NotImplementedError

    def inject_noise(self, input: torch.tensor):
        # change quantization range to unsigned integer
        x_uint_flat = self.int2bin(input).reshape(-1)
        # sample noise
        epsilon = sample_noise(self.p0, shape=(len(x_uint_flat), self.N_bits), device=input.device)
        # inject noise
        x_uint_pertub_flat = x_uint_flat.clone().short() ^ epsilon
        # return quantization range to signed integer
        x_int_pertub_flat = self.bin2int(x_uint_pertub_flat)
        x_int_pertub = x_int_pertub_flat.reshape(input.shape)

        return (x_int_pertub + input) - input.detach()

    def forward(self, input: torch.tensor):
        if self.N_bits == 32:
            return input
        else:
            # for monitoring weights
            self.max_range = input.max().item()
            self.min_range = input.min().item()
            input_int = self.linear_quantize(input)
            if self.p0 > 0:
                input_int = self.inject_noise(input_int)
            output = self.linear_dequantize(input_int)
            return output

    def int2bin(self, weight_int: torch.tensor):
        if self.signed:
            weight_int = int2bin(weight_int, self.N_bits).short()
        else:
            weight_int = weight_int.short()
        return weight_int

    def bin2int(self, weight_uint: torch.tensor):
        if self.signed:
            weight = bin2int(weight_uint, self.N_bits).float()
        else:
            weight = weight_uint.float()
        return weight


@Quantizer.register("normal")
class Normal(Quantizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_size = torch.tensor(1.)
        self.zero_point = torch.tensor(0.)

    def linear_quantize(self, input: torch.tensor):
        self.step_size = (input.abs().max() / (2 ** (self.N_bits - 1))).detach()
        x = torch.clamp((input / self.step_size) - self.zero_point, self.Qn, self.Qp)
        #x = round_pass(x)
        x = (x.round() - (input / self.step_size)).detach() + (input / self.step_size)
        return x

    def linear_dequantize(self, input: torch.tensor):
        return (input + self.zero_point) * self.step_size

    def _init_q_params(self, input: torch.tensor):
        pass

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range}


@Quantizer.register("lsq")
class LSQ(Quantizer):
    def __init__(self, use_grad_scaled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_grad_scaled = use_grad_scaled
        self.step_size = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.zero_point = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def linear_quantize(self, input: torch.tensor):
        if self.use_grad_scaled:
            s_grad_scale = 1.0 / ((self.Qp * input.numel()) ** 0.5)
            step_size = grad_scale(self.step_size, s_grad_scale)
        else:
            step_size = self.step_size
        x = torch.clamp((input / step_size) - self.zero_point, self.Qn, self.Qp)
        x = round_pass(x)
        return x

    def linear_dequantize(self, input: torch.tensor):
        return (input + self.zero_point) * self.step_size

    def _init_q_params(self, input: torch.tensor):
        self.step_size.data = input.detach().abs().mean() * 2 / ((2 ** (self.N_bits - 1) - 1) ** 0.5)

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range,
                'range_pos': (self.step_size*self.Qp).item(), 'range_neg': (self.step_size*self.Qn).item()}


@Quantizer.register("wcat")
class WCAT(Quantizer):
    def __init__(self, clip: _Clamp, use_grad_scaled: bool = True, init_method: str = 'max_abs',
                 noisy_mse_ber: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip
        self.use_grad_scaled = use_grad_scaled
        self.init_method = init_method
        self.noisy_mse_ber = noisy_mse_ber
        self.q_range = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.step_size = torch.tensor(1.)
        self.zero_point = torch.tensor(0.)

    def linear_quantize(self, input: torch.tensor):
        if self.use_grad_scaled:
            q_range_grad_scale = 1.0 / (input.numel() ** 0.5)
            q_range = grad_scale(self.q_range, q_range_grad_scale)
        else:
            q_range = self.q_range
        x = self.clip.apply(input, q_range, self.signed)
        if self.signed:
            self.step_size = q_range.detach() / (2 ** (self.N_bits - 1))
        else:
            self.step_size = q_range.detach() / (2 ** self.N_bits - 1)
        x_int = round_pass((x / self.step_size) - self.zero_point)
        x_clip = torch.clamp(x_int, self.Qn, self.Qp)
        return (x_clip - x_int).detach() + x_int

    def linear_dequantize(self, input: torch.tensor):
        return (input + self.zero_point) * self.step_size

    def _init_q_params(self, input: torch.tensor):
        if self.init_method == 'max_abs':
            self.q_range.data = input.detach().abs().max()
        elif self.init_method == 'SAWB':
            self.q_range.data = get_scale(input, self.N_bits)
        elif self.init_method == 'MSE':
            self.q_range.data = self._bruteforce_optimal_MSE(input)
        elif self.init_method == 'noisy_MSE':
            self.q_range.data = self._bruteforce_optimal_MSE_faulty(input)
        else:
            raise NotImplementedError

    def _bruteforce_optimal_MSE(self, input: torch.tensor):

        def mse(q_range_star: float = 1.):
            self.q_range.data = torch.tensor(q_range_star)
            input_quant = self.forward(input)
            return torch.nn.functional.mse_loss(input, input_quant).item()

        # q_range_ = np.array(max(input.abs().max().detach(), 1e-10))
        q_range_ = np.array(input.detach().abs().mean() * 2 * ((2 ** (self.N_bits - 1) - 1) ** 0.5))
        res = minimize(mse, q_range_, method='Nelder-Mead', tol=1e-6)
        assert res.success
        return torch.tensor(res.x[0])

    def _bruteforce_optimal_MSE_faulty(self, input: torch.tensor):
        # TODO: only one noise realization is used, should change to random noise for each bit
        def mse_noisy(q_range_star: float = 1.):
            self.q_range.data = torch.tensor(q_range_star)
            input_int = self.linear_quantize(input)
            if self.signed:
                input_int = int2bin(input_int, self.N_bits).short()
            bw = 2 ** torch.arange(self.N_bits).short()
            expected_MSE = 0
            for i in range(2 ** self.N_bits):
                epsilon = binary(torch.tensor(i), self.N_bits).short()
                prob_epsilon = (epsilon * self.noisy_mse_ber + (1 - epsilon) * (1 - self.noisy_mse_ber)).prod()
                epsilon = epsilon @ bw
                input_int_pertub = input_int ^ epsilon.to(input.device)
                if self.signed:
                    input_int_pertub = bin2int(input_int_pertub, self.N_bits).float()
                else:
                    input_int_pertub = input_int_pertub.float()
                input_pertub = self.linear_dequantize(input_int_pertub)
                expected_MSE += torch.nn.functional.mse_loss(input, input_pertub).item() * prob_epsilon

            return expected_MSE

        # q_range_ = np.array(max(input.abs().max().detach(), 1e-10))
        q_range_ = np.array(input.detach().abs().mean() * 2 * ((2 ** (self.N_bits - 1) - 1) ** 0.5))
        res = minimize(mse_noisy, q_range_, method='Nelder-Mead', tol=1e-6)
        assert res.success
        return torch.tensor(res.x[0])

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range,
                'range_pos': self.q_range.item(), 'range_neg': (-self.q_range).item()}


@Quantizer.register("wclip")
class WClip(Quantizer):
    def __init__(self, wclip: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = nn.Parameter(torch.tensor(wclip), requires_grad=False)
        self.step_size = torch.tensor(1.)
        self.zero_point = torch.tensor(0.)

    def linear_quantize(self, input: torch.tensor):

        x = torch.clamp(input, min=-self.alpha.detach(), max=self.alpha.detach())
        if self.signed:
            self.step_size = self.alpha.detach() / (2 ** (self.N_bits - 1))
        else:
            self.step_size = self.alpha.detach() / (2 ** self.N_bits - 1)
        x_int = round_pass((x / self.step_size) - self.zero_point)
        x_clip = torch.clamp(x_int, self.Qn, self.Qp)
        return (x_clip - x_int).detach() + x_int

    def linear_dequantize(self, input: torch.tensor):
        return (input + self.zero_point) * self.step_size

    def _init_q_params(self, input: torch.tensor):
        pass

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range}

@Quantizer.register("plclip")
class PLClip(Quantizer):
    def __init__(self, plclip: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.plclip = plclip
        self.alpha = nn.Parameter(torch.tensor(plclip), requires_grad=False)
        self.step_size = torch.tensor(1.)
        self.zero_point = torch.tensor(0.)

    def linear_quantize(self, input: torch.tensor):

        x = torch.clamp(input, min=-self.alpha.detach(), max=self.alpha.detach())
        if self.signed:
            self.step_size = self.alpha.detach() / (2 ** (self.N_bits - 1))
        else:
            self.step_size = self.alpha.detach() / (2 ** self.N_bits - 1)
        x_int = round_pass((x / self.step_size) - self.zero_point)
        x_clip = torch.clamp(x_int, self.Qn, self.Qp)
        return (x_clip - x_int).detach() + x_int

    def linear_dequantize(self, input: torch.tensor):
        return (input + self.zero_point) * self.step_size

    def set_alpha(self, adapted_alpha):
        self.alpha.data = torch.tensor(adapted_alpha).to(self.alpha.device)

    def _init_q_params(self, input: torch.tensor):
        pass

    def monitor_ranges(self):
        return {'max_weight': self.max_range, 'min_weight': self.min_range,
                'range_pos': self.alpha.item(), 'range_neg': (-self.alpha).item()}


class Quantized_Linear(nn.Linear):
    def __init__(self, weight_quantize_module: Quantizer, in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        for name, p in weight_quantize_module.named_parameters():
            self.register_parameter(name='weight_' + name, param=p)

    def forward(self, input):
        # TODO: activation quantization
        weight_quant = self.weight_quantize_module(self.weight)
        return F.linear(input, weight_quant, self.bias)


class Quantized_Conv2d(nn.Conv2d):
    def __init__(self, weight_quantize_module: Quantizer, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(Quantized_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias)
        self.weight_quantize_module = weight_quantize_module
        for name, p in weight_quantize_module.named_parameters():
            self.register_parameter(name='weight_' + name, param=p)

    def forward(self, input):
        # TODO: activation quantization
        weight_quant = self.weight_quantize_module(self.weight)
        return F.conv2d(input, weight_quant, self.bias, self.stride, self.padding, self.dilation, self.groups)

