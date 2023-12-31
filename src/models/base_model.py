import torch
import torch.nn as nn
import wandb
from quantization import Quantizer, Quantized_Linear, Quantized_Conv2d
from common import FromParams, Registrable, Params, Lazy
import os
import logging
from typing import Iterator


def make_quant_layer(weight_quantize_module: Quantizer, module: torch.nn.Module, layer_type: torch.nn):
    """
    code modified from: https://github.com/IST-DASLab/gptq
    recursively replace the layers of the model with `layer_type`, with quantized version one
    Args:
        weight_quantize_module: quantizer module
        module: model
        layer_type: nn.Linear or nn.Conv2d

    Returns:

    """
    if isinstance(module, Quantized_Conv2d) or isinstance(module, Quantized_Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) == layer_type:
            setattr(
                module, attr,
                construct_quant_layer(weight_quantize_module, tmp, layer_type)
            )
            quant_layer = module.__getattr__(attr)
            quant_layer.weight.data = tmp.weight.data
            if tmp.bias is not None:
                quant_layer.bias.data = tmp.bias.data

    for name1, child in module.named_children():
        make_quant_layer(weight_quantize_module, child, layer_type)


def construct_quant_layer(weight_quantize_module: Quantizer, layer: torch.nn.Module, layer_type: torch.nn):
    wq_module = weight_quantize_module.construct()
    wq_module._init_q_params(layer.weight)
    if layer_type == nn.Linear:
        return Quantized_Linear(wq_module, layer.in_features, layer.out_features,
                                layer.bias is not None)
    elif layer_type == nn.Conv2d:
        return Quantized_Conv2d(wq_module, layer.in_channels, layer.out_channels,
                                layer.kernel_size,
                                stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                groups=layer.groups,
                                bias=layer.bias is not None)


class Base_Model(Registrable):
    def __init__(self, weight_quantize_module: Lazy[Quantizer], exp_name: str = None,
                 save_path: str = './save', device: str = 'cuda'):
        #super().__init__()
        self.weight_quantize_module = weight_quantize_module
        self.exp_name = exp_name
        self.save_path = save_path
        if not torch.cuda.is_available() and 'cuda' in device:
            logging.warning('CUDA not available')
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        if len(weight_quantize_module._params) > 0:
            self._model2quant()
        self.to(self.device)

    def _model2quant(self):
        make_quant_layer(self.weight_quantize_module, self, nn.Linear)
        make_quant_layer(self.weight_quantize_module, self, nn.Conv2d)

    def _set_plclip_values(self):
        if self.weight_quantize_module._params.get('type') == 'plclip':
            plclip = self.weight_quantize_module._params.get('plclip')
            with torch.no_grad():
                weight_max_list = []
                for name, m in self.named_modules():
                    if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                        weight_max_list.append(m.weight.abs().max().detach())
                for name, m in self.named_modules():
                    if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                        kl = m.weight.abs().max().detach() / max(weight_max_list)
                        plclip_ = max(0.2, kl) * plclip
                        m.weight_quantize_module.set_alpha(plclip_)

    def compute_max_magnitude_loss(self):
        loss = torch.tensor(0.).to(self.device)
        for name, m in self.named_modules():
            if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                if m.bias is not None:
                    loss += (m.weight.abs().max() + m.bias.abs().max())
                else:
                    loss += m.weight.abs().max()
        return loss

    def modified_forward(self, input):
        self._set_plclip_values()
        return self.forward(input)

    def monitoring_range(self, wandb_logs=False):
        items = {}
        for name, m in self.named_modules():
            if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                res = m.weight_quantize_module.monitor_ranges()
                for key, value in res.items():
                    items[f'{name}_{key}'] = value
        if wandb_logs:
            wandb.log(items)
        logging.info(items)
