import os
import torch
import copy
from quantization import Quantized_Linear, Quantized_Conv2d
from train import evaluate
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import logging
import wandb
import numpy as np
import csv
from .attack_base import Attack


def generate_noise_mask(m, p0, noise_type='random'):
    """
        Input
            m: Gumbel_Conv2d or Gumbel_Linear module
            p0: bit error rate
            noise_type: injecting noise randomly,only to MSB or ...
        Output
            mask: binary tensor with shape of (numel, N_bit). '1' indicate a bit flip
            weight_int_flat: flat quantized weight
        """
    weight_int = m.weight_quantize_module.linear_quantize(m.weight)
    weight_uint_flat = m.weight_quantize_module.int2bin(weight_int).reshape(-1)

    # generate noise mask
    if noise_type == 'random':
        # randomly select bits
        # mask = torch.bernoulli(torch.ones(size=(len(bin_w), m.N_bits)) * p0).to(torch.int).to(bin_w.device)
        uniform = torch.distributions.Uniform(torch.tensor([0.]), torch.tensor([1.]))
        unif_sample = uniform.sample((len(weight_uint_flat), m.weight_quantize_module.N_bits))[:, :, 0]
        mask = ((unif_sample < p0) * 1).to(torch.int).to(weight_uint_flat.device)
    elif noise_type == 'MSB':
        mask_0 = torch.bernoulli(torch.ones(size=(1, len(weight_uint_flat))) * p0).to(torch.int)
        mask_1 = torch.zeros(m.weight_quantize_module.N_bits - 1, len(weight_uint_flat)).to(torch.int)
        mask = torch.concat((mask_0, mask_1), 0)
        mask = mask.flip(0).T
    else:
        raise "noise type couldn't find"

    mask = mask.to(weight_uint_flat.device)
    return mask, weight_uint_flat


def generate_protect_index(mask, weight_bits, protect_bit_index='NoProtection',
                           protect_bit_amount=0.25):
    """

    Args:
        mask: noise mask
        weight_bits: number of bits
        protect_bit_index: choose between No Protection, MSB, ...
        protect_bit_amount: the ratio of number of protected bits over total bits.

    Returns:
        mask: updated noise mask

    """
    # generate protect index
    if protect_bit_index == 'NoProtection':
        return mask
    elif protect_bit_index == 'MSB':
        protected_bit_idx_0 = torch.bernoulli(torch.ones(size=(1, len(mask))) * protect_bit_amount).to(torch.int)
        protected_bit_idx_1 = torch.zeros(weight_bits - 1, len(mask)).to(torch.int)
        protected_bit_idx = torch.concat((protected_bit_idx_0, protected_bit_idx_1), 0)
        protected_bit_idx = protected_bit_idx.flip(0).T
        protected_bit_idx = protected_bit_idx.to(mask.device)
        mask = mask * protected_bit_idx
        return mask
    elif protect_bit_index == 'oracle':
        raise NotImplementedError
    else:
        raise "Invalid protect index"


def apply_noise(m, p0, noise_type, protect_bit_index, protect_bit_amount):
    """

    Args:
        m: nn.Module
        p0: bit error rate
        noise_type: injecting noise randomly,only to MSB or ...
        protect_bit_index: choose between No Protection, MSB, ...
        protect_bit_amount: the ratio of number of protected bits over total bits.

    Returns:
    """
    mask, weight_uint_flat = generate_noise_mask(m, p0, noise_type)
    mask = generate_protect_index(mask, m.weight_quantize_module.N_bits, protect_bit_index, protect_bit_amount)
    bw = 2 ** torch.arange(m.weight_quantize_module.N_bits).to(torch.int).to(weight_uint_flat.device)
    mask = torch.sum(mask * bw, 1)
    weight_uint_flat_pertub = weight_uint_flat.clone() ^ mask
    weight_int_flat_pertub = m.weight_quantize_module.bin2int(weight_uint_flat_pertub)
    weight_flat_pertub = m.weight_quantize_module.linear_dequantize(weight_int_flat_pertub)
    m.weight.data = weight_flat_pertub.view(m.weight.data.size())

@Attack.register('rbf')
class RBF(Attack):
    def __init__(self, noise_type: str = 'random', protect_bit_index: str = 'NoProtection',
                 protect_bit_amount: float = 0.25, module_list: list = None, attack_MC_sample: int = 50,
                 wandb_logs: bool = False, save_path: str = './save/test.csv', **kwargs):
        Attack.__init__(self, **kwargs)
        self.noise_type = noise_type
        self.protect_bit_index = protect_bit_index
        self.protect_bit_amount = protect_bit_amount
        self.attack_MC_sample = attack_MC_sample
        self.wandb_logs = wandb_logs
        self.save_path = save_path
        self.csv_col = ["BER", "acc_mean", "acc_std", "acc_min", "acc_max"]
        self.result_dict = {"BER": [], "acc_mean": [], "acc_std": [], "acc_min": [], "acc_max": []}
        self.module_list = module_list

    def simulate(self, model, p0, test_loader):
        if self.module_list is None:
            module_list = []
            for name, m in model.named_modules():
                if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                    module_list.append(name)
        else:
            module_list = self.module_list

        if p0 == 0:
            test_loss, test_acc = evaluate(model, test_loader)
            logging.info(f"clean model--> test accuracy: {test_acc}%, test loss: {test_loss}")
            self.result_dict["BER"].append(0)
            self.result_dict["acc_mean"].append(test_acc)
            self.result_dict["acc_min"].append(test_acc)
            self.result_dict["acc_max"].append(test_acc)
            self.result_dict["acc_std"].append(0)
            if self.wandb_logs:
                wandb.log({"BER": 0, "acc_mean": test_acc,
                           "acc_std": 0, "acc_min": test_acc,
                           "acc_max": test_acc})

        else:
            test_accuracy = []
            with logging_redirect_tqdm():
                for i in trange(self.attack_MC_sample):
                    model_copy = copy.deepcopy(model)
                    for name, m in model_copy.named_modules():
                        if name in module_list:
                            apply_noise(m, p0, self.noise_type, self.protect_bit_index, self.protect_bit_amount)
                    test_loss, test_acc = evaluate(model_copy, test_loader)
                    test_accuracy.append(test_acc)
            logging.info(f"BER:{p0}, acc_mean:{np.mean(test_accuracy)}, acc_std:{np.std(test_accuracy)}, "
                         f"acc_min:{np.min(test_accuracy)}, acc_max:{np.max(test_accuracy)}")
            if self.wandb_logs:
                wandb.log({"BER": p0, "acc_mean": np.mean(test_accuracy),
                           "acc_std": np.std(test_accuracy), "acc_min": np.min(test_accuracy),
                           "acc_max": np.max(test_accuracy)})
            self.result_dict["BER"].append(p0)
            self.result_dict["acc_mean"].append(np.mean(test_accuracy))
            self.result_dict["acc_min"].append(np.min(test_accuracy))
            self.result_dict["acc_max"].append(np.max(test_accuracy))
            self.result_dict["acc_std"].append(np.std(test_accuracy))

        with open(self.save_path, "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.csv_col)
            writer.writerows(zip(*self.result_dict.values()))
