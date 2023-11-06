import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
import json
import _jsonnet
import torch
from quantization import Quantized_Linear, Quantized_Conv2d


def weight_histogram(model, save_path, exp_name, modules_name=None, wandb_=False):
    if modules_name is None:
        name_list = [name for name, m in model.named_modules() if
                     isinstance(m, Quantized_Linear) or isinstance(m, Quantized_Conv2d)]
    else:
        name_list = modules_name

    fig, axes = plt.subplots(2, len(name_list), figsize=(len(name_list) * 5, 2 * 3))
    j = 0
    for name, m in model.named_modules():
        if name in name_list:
            if m.weight_quantize_module.N_bits > 8:
                real_weight = m.weight.reshape(-1).detach().cpu()
                num_bins = 16
                counts, bins = np.histogram(real_weight, num_bins)
                width_ = (max(bins) - min(bins)) / num_bins
                counts = counts / real_weight.numel()
                axes[0, j].bar(bins[:-1], counts, width=width_, alpha=0.4, label='real')
                axes[0, j].set_title(name)
                axes[0, j].grid()
                axes[0, j].legend()

            else:
                weight_quant = m.weight_quantize_module(m.weight)
                weight = weight_quant.reshape(-1).detach().cpu().numpy()
                unique_weight = np.unique(weight)
                density = np.zeros_like(unique_weight)
                for i in range(len(density)):
                    density[i] = np.isclose(weight, np.ones_like(weight) * unique_weight[i]).sum()
                density = density / len(weight)
                width = (max(unique_weight) - min(unique_weight)) / len(unique_weight)

                real_weight = m.weight.reshape(-1).detach().cpu()
                num_bins = max(16, len(unique_weight))
                counts, bins = np.histogram(real_weight, num_bins)
                width_ = (max(bins) - min(bins)) / num_bins
                counts = counts / real_weight.numel()
                axes[0, j].bar(bins[:-1], counts, width=width_, label='real')
                axes[0, j].set_title(name)
                axes[0, j].grid()
                axes[0, j].legend()

                axes[1, j].bar(unique_weight, density, width=width, color='darkorange', label='quantized')
                axes[1, j].grid()
                axes[1, j].legend()

            j += 1
    fig.tight_layout()
    fig_path = os.path.join(save_path, exp_name)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    plt.savefig(os.path.join(fig_path, f'{exp_name}.png'), dpi=200)
    if wandb_:
        wandb.save(fig_path)


def weight_violinplot(model, save_path, exp_name, modules_name=None, wandb_=False):
    if modules_name is None:
        name_list = [name for name, m in model.named_modules() if
                     isinstance(m, Quantized_Linear) or isinstance(m, Quantized_Conv2d)]
    else:
        name_list = modules_name
    real_weights = []
    weight_quants = []
    for name, m in model.named_modules():
        if name in name_list:
            if m.weight_bits is None or m.weight_bits > 8:
                real_weight = m.weight.reshape(-1).detach().cpu().numpy()
                real_weights.append(real_weight)
            else:
                weight_quant = m.weight_quantize_module(m.weight).reshape(-1).detach().cpu().numpy()
                real_weight = m.weight.reshape(-1).detach().cpu().numpy()
                real_weights.append(real_weight)
                weight_quants.append(weight_quant)
    if len(weight_quants) > 0:
        xticks = np.arange(len(name_list))
        fig, axes = plt.subplots(2, 1, figsize=(len(name_list) * 1., 10))
        axes[0].violinplot(real_weights)
        axes[0].set_xticks([])
        axes[0].grid()
        axes[0].set_title(exp_name)
        parts = axes[1].violinplot(weight_quants)
        for pc in parts['bodies']:
            pc.set_color('darkorange')
        parts['cmaxes'].set_color('darkorange')
        parts['cmins'].set_color('darkorange')
        parts['cbars'].set_color('darkorange')
        axes[1].grid()
        plt.xticks(np.arange(len(name_list)) + 1.1, name_list, rotation=65, fontsize=12)
    else:
        xticks = np.arange(len(name_list))
        fig = plt.figure(figsize=(len(name_list) * 1., 4))
        plt.violinplot(real_weights)
        plt.grid()
        plt.xticks(np.arange(len(name_list)) + 1.1, name_list, rotation=65, fontsize=12)
        plt.title(exp_name)
    fig.tight_layout()
    fig_path = os.path.join(save_path, exp_name)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    plt.savefig(os.path.join(fig_path, f'{exp_name}.png'), dpi=200)
    if wandb_:
        wandb.save(fig_path)


def plot_monitor_range(experiment, wandb_run_path: str = None):

    def remove_None(arr):
        new_arr = [_ for _ in arr if _ is not None]
        return new_arr

    model = experiment.model
    cfg_path = os.path.join(experiment.save_path, experiment.exp_name, 'config.jsonnet')
    json_str = _jsonnet.evaluate_file(cfg_path)
    cfg = json.loads(json_str)
    if 'wandb_run_path' in cfg.keys():
        wandb_run_path = cfg['wandb_run_path']
    api = wandb.Api()
    run = api.run(wandb_run_path)
    metrics_scan_history = run.scan_history()
    metrics_dataframe = {}
    for key in run.history().keys():
        metrics_dataframe[key] = []

    for i, dict_iter in enumerate(metrics_scan_history):
        for key in dict_iter.keys():
            metrics_dataframe[key].append(dict_iter[key])

    for key in metrics_dataframe.keys():
        metrics_dataframe[key] = remove_None(metrics_dataframe[key])
    layer_name_list = []
    for name, m in model.named_modules():
        if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
            layer_name_list.append(name)

    monitored_values = model.weight_quantize_module.construct().monitor_ranges().keys()
    fig, axes = plt.subplots(1, len(layer_name_list), figsize=(len(layer_name_list) * 4, 1 * 5))
    j = 0
    for name in layer_name_list:
        if len(monitored_values) > 2:
            x_bar = torch.arange(len(metrics_dataframe[f"{name}_max_weight"]))
            axes[j].plot(x_bar, metrics_dataframe[f"{name}_max_weight"], label='max_weight', color='b', alpha=0.7)
            axes[j].plot(x_bar, metrics_dataframe[f"{name}_min_weight"], label='min_weight', color='b', alpha=0.7)
            axes[j].plot(x_bar, metrics_dataframe[f"{name}_range_pos"], label='range_pos', color='r', alpha=0.9)
            axes[j].plot(x_bar, metrics_dataframe[f"{name}_range_neg"], label='range_neg', color='r', alpha=0.9)
            axes[j].set_title(name)
            axes[j].legend()
            axes[j].grid()
        else:
            x_bar = torch.arange(len(metrics_dataframe[f"{name}_max_weight"]))
            axes[j].plot(x_bar, metrics_dataframe[f"{name}_max_weight"], label='max_weight', color='b', alpha=0.7)
            axes[j].plot(x_bar, metrics_dataframe[f"{name}_min_weight"], label='min_weight', color='b', alpha=0.7)
            axes[j].set_title(name)
            axes[j].legend()
            axes[j].grid()
        j += 1
    fig.suptitle(run.name, fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(experiment.save_path, experiment.exp_name, 'monitor_range.png')
    plt.savefig(fig_path, dpi=200)
