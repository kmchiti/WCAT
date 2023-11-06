"""
Modified from https://github.com/elliothe/Neural_Network_Weight_Attack
"""
import copy
import random
import torch
import numpy as np
import operator
import pandas as pd
import os
import wandb
import warnings

from .attack_base import Attack
from quantization import Quantizer, Quantized_Linear, Quantized_Conv2d, binary, hamming_distance


@Attack.register('bfa')
class BFA(Attack):
    def __init__(self, criterion, k_top, wandb_logs: bool = False, attack_sample_size: int = 50,
                 save_path: str = './save/test.csv', **kwargs):
        Attack.__init__(self, **kwargs)
        self.wandb_logs = wandb_logs
        self.save_path = save_path
        self.attack_sample_size = attack_sample_size

        self.loss_max = None
        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        # attributes for random attack
        self.module_list = []
        self.idx_dict = {}
        self.attack_logs = {'num_bit_flip': [], 'module': [], 'weight_idx': [], 'weight_prior_': [],
                            'weight_post_': [], 'attack_mask': [], 'hamming_distance': []}

    def flip_bit(self, module, name, weight_int):
        """
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        """
        b_w = 2 ** torch.arange(start=module.weight_quantize_module.N_bits - 1, end=-1, step=-1).unsqueeze(-1).float()
        b_w = b_w.to(module.weight.device)
        b_w[0] = -b_w[0]
        if self.k_top is None:
            k_top = module.weight.detach().flatten().__len__()
        else:
            k_top = self.k_top
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = module.weight.grad.detach().abs().view(-1).topk(k_top)
        # update the b_grad to its signed representation
        w_grad_topk = module.weight.grad.detach().view(-1)[w_idx_topk]

        # 2. create the b_grad matrix in shape of [weight_bits, k_top]
        b_grad_topk = w_grad_topk * b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        weight_uint_flat = module.weight_quantize_module.int2bin(weight_int).reshape(-1)
        w_bin_topk = weight_uint_flat[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = torch.div(
            (w_bin_topk.repeat(module.weight_quantize_module.N_bits, 1) & b_w.abs().repeat(1, k_top).short()),
            b_w.abs().repeat(1, k_top).short(), rounding_mode='floor')
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()

        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.weight_bits2flip
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()

        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            warnings.warn(f"all grad zin {name} is zero!! random attack is applied")
            b_grad_max_idx = torch.randint(0, len(bit2flip), (1,))
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())

        # 6. Based on the identified bit indexed by ```bit2flip```, generate another
        # mask, then perform the bitwise xor operation to realize the bit-flip.
        w_bin_topk_flipped = (bit2flip.short() * b_w.abs().short()).sum(0, dtype=torch.int16) ^ w_bin_topk

        # 7. update the weight in the original weight tensor
        weight_uint_flat[w_idx_topk] = w_bin_topk_flipped  # in-place change
        weight_int_flat_pertub = module.weight_quantize_module.bin2int(weight_uint_flat)
        weight_flat_pertub = module.weight_quantize_module.linear_dequantize(weight_int_flat_pertub)
        weight_pertub = weight_flat_pertub.view(module.weight.data.size()).float()

        return weight_pertub

    def progressive_bit_search(self, model, data, target):
        """
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped.
        """

        for name, m in model.named_modules():
            if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                self.module_list.append(name)
                self.idx_dict[name] = []

        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()

        # 1. perform the inference w.r.t given data and target
        output = model(data.to(model.device))
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target.to(model.device))
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # 3. for each layer flip #bits = self.bits2flip
        while self.loss_max <= self.loss.item():
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            for name, module in model.named_modules():
                if isinstance(module, Quantized_Conv2d) or isinstance(module, Quantized_Linear):
                    clean_weight = module.weight.data.detach()
                    weight_int = module.weight_quantize_module.linear_quantize(module.weight)

                    attack_weight = self.flip_bit(module, name, weight_int)
                    # change the weight to attacked weight and get loss
                    module.weight.data = attack_weight
                    output = model(data.to(model.device))
                    self.loss_dict[name] = self.criterion(output, target.to(model.device)).item()
                    # change the weight back to the clean weight
                    module.weight.data = clean_weight

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = max(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change that layer's weight without putting back the clean weight
        for module_idx, (name, module) in enumerate(model.named_modules()):
            if name == max_loss_module:
                weight_int = module.weight_quantize_module.linear_quantize(module.weight)
                attack_weight = self.flip_bit(module, name, weight_int)
                attack_weight_int = module.weight_quantize_module.linear_quantize(attack_weight)
                #############################################
                ## Attack profiling
                #############################################
                weight_int = weight_int.short()
                attack_weight_int = attack_weight_int.short()

                weight_mismatch = attack_weight_int - weight_int
                attack_weight_idx = torch.nonzero(weight_mismatch)

                self.attack_logs['num_bit_flip'].append(self.bit_counter + 1)
                self.attack_logs['module'].append(max_loss_module)
                self.attack_logs['weight_idx'].append(list(attack_weight_idx[0].detach().cpu().numpy()))
                self.attack_logs['weight_prior_'].append(module.weight.detach()[list(attack_weight_idx[0])].item())
                self.attack_logs['weight_post_'].append(attack_weight.clone()[list(attack_weight_idx[0])].item())
                attack_mask = weight_int[list(attack_weight_idx[0])] ^ attack_weight_int[list(attack_weight_idx[0])]
                attack_mask = module.weight_quantize_module.int2bin(attack_mask)
                self.attack_logs['attack_mask'].append(
                    str(list(binary(attack_mask, module.weight_quantize_module.N_bits).detach().cpu().numpy())))
                self.attack_logs['hamming_distance'].append(
                    hamming_distance(weight_int, attack_weight_int, module.weight_quantize_module.N_bits))

                ###############################################################

                module.weight.data = attack_weight

        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

    def save_attack_logs(self):
        df = pd.DataFrame.from_dict(self.attack_logs)
        df.to_csv(self.save_path)
        if self.wandb_logs:
            wandb.log({"attack_logs": wandb.Table(dataframe=df)})

