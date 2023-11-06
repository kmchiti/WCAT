import os
import random
import numpy as np
import torch
import wandb
import logging

import json
import copy
from models import Base_Model
from data import Dataset
from train import Trainer, evaluate
from common import FromParams, Lazy, Params
from plots import weight_histogram, plot_monitor_range
from attack import Attack, hamming_distance_model
from quantization import *
from typing import Dict, Any


class Runtime(FromParams):
    def __init__(self, seed: int, _project_name: str, _entity: str, model: Lazy[Base_Model], trainer: Lazy[Trainer],
                 dataset: Lazy[Dataset], attack: Lazy[Attack], _save_path: str = './save', _wandb_logs: bool = False,
                 _resume: str = None):
        self.model = model
        self.trainer = trainer
        self.dataset = dataset
        self.attack = attack
        self.seed = seed
        self.project_name = _project_name
        self.entity = _entity
        self.save_path = _save_path
        self.wandb_logs = _wandb_logs
        self.exp_name = None
        self.resume = _resume
        os.makedirs(_save_path, exist_ok=True)

    def setup(self, EXPERIMENT_NAME: str, cfg: Dict[str, Any]):
        self.set_seed()
        self.exp_name = EXPERIMENT_NAME
        self.model = self.model.construct(exp_name=self.exp_name, save_path=self.save_path)
        self.trainer = self.trainer.construct(exp_name=self.exp_name, save_path=self.save_path,
                                              wandb_logs=self.wandb_logs)
        self.dataset = self.dataset.construct()
        os.makedirs(os.path.join(self.save_path, EXPERIMENT_NAME), exist_ok=True)
        self.setup_logging(log_path=os.path.join(self.save_path, EXPERIMENT_NAME))
        cfg = self.setup_wandb(cfg)

        jsonnet_string = json.dumps(cfg, indent=4)
        save_path = os.path.join(self.save_path, self.exp_name, 'config.jsonnet')
        with open(save_path, 'w') as jsonnet_file:
            jsonnet_file.write(jsonnet_string)
        logging.info(f'configuration file saved at: {save_path}')

    def setup_wandb(self, cfg: Dict[str, Any]):
        if self.wandb_logs:
            wandb.init(project=self.project_name, entity=self.entity, name=self.exp_name, config=cfg)
            cfg['wandb_run_path'] = wandb.run.path
        return cfg

    def set_seed(self):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

    @staticmethod
    def setup_logging(log_path):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_path, 'logfile.log'))
        file_handler.setLevel(logging.INFO)  # Set the desired logging level
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(file_handler)

    def load_model(self):
        if self.resume is not None:
            # load checkpoint to resume training
            model_path = os.path.join(self.save_path, self.exp_name, f'{self.resume}')
            logging.info(f"=> loading checkpoint from: {model_path}")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.model.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                resume_epoch = checkpoint['epoch']
                return resume_epoch
            else:
                logging.warning(f"model_path: {model_path} didn't exist")
                raise "path to model didn't exist."
        else:
            model_path = os.path.join(self.save_path, self.exp_name, 'model_checkpoint.pth')
            logging.info(f"=> loading checkpoint from: {model_path}")
            if os.path.exists(model_path):
                loaded_state_dict = torch.load(model_path, map_location=self.model.device)
                network_kvpair = self.model.state_dict()
                for key in loaded_state_dict.keys():
                    network_kvpair[key] = loaded_state_dict[key]
                self.model.load_state_dict(network_kvpair)
            else:
                logging.warning(f"model_path: {model_path} didn't exist")
                raise "path to model didn't exist."

    def evaluate(self):
        train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
        test_loss, test_acc = evaluate(self.model, test_loader)
        if self.wandb_logs:
            wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})
        logging.info(f"test accuracy: {test_acc}%, test loss: {test_loss}")

    def load_and_evaluate(self):
        self.load_model()
        train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
        test_loss, test_acc = evaluate(self.model, test_loader)
        if self.wandb_logs:
            wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})
        logging.info(f"test accuracy: {test_acc}%, test loss: {test_loss}")

    def train(self):
        self.trainer.build(self.model)
        if hasattr(self.dataset, 'use_train'):
            self.dataset.use_train = True
        train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
        self.trainer.fit(self.model, train_loader, valid_loader)

    def resume_and_train(self):
        self.trainer.build(self.model)
        resume_epoch = self.load_model()
        if hasattr(self.dataset, 'use_train'):
            self.dataset.use_train = True
        train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
        self.trainer.fit(self.model, train_loader, valid_loader, resume_epoch=resume_epoch)

    def random_flip(self):
        result_path = os.path.join(self.save_path, self.exp_name, 'random_flip_result.csv')
        self.attack = self.attack.construct(wandb_logs=self.wandb_logs, save_path=result_path)

        self.load_model()
        train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
        BER_list = np.array([0., 0.0001, 0.0004, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01])
        # BER_list = np.array([0.0001, 0.0005, 0.001, 0.005])
        for ber in BER_list:
            self.attack.simulate(self.model, ber, test_loader)

    def bfa(self):

        result_path = os.path.join(self.save_path, self.exp_name, 'BFA_result.csv')
        self.attack = self.attack.construct(criterion=self.trainer.criterion, wandb_logs=self.wandb_logs,
                                            save_path=result_path)

        # fix the range for normal, PLClip and WClip by changing them to WCAT
        if isinstance(self.model.weight_quantize_module, PLClip) or \
                isinstance(self.model.weight_quantize_module, WClip) or \
                isinstance(self.model.weight_quantize_module, Normal):
            config = {'N_bits': self.model.weight_quantize_module.N_bits, 'signed': 1, 'p0': 0.0,
                      'type': "wcat", 'use_grad_scaled': 1, 'init_method': "max_abs",
                      'clip': {'type': "ste", }}
            self.model.weight_quantize_module = Lazy[Quantizer.from_params(Params(config))]
        for name, m in self.model.named_modules():
            if isinstance(m, Quantized_Conv2d) or isinstance(m, Quantized_Linear):
                m.weight_quantize_module = self.model.weight_quantize_module.construct()

        self.dataset.attack_sample_size = self.attack.attack_sample_size
        train_loader, test_loader, valid_loader, calibration_loader = self.dataset.build()
        self.load_model()
        model = copy.deepcopy(self.model)
        test_loss, test_acc = evaluate(model, test_loader)
        logging.info(f"clean model--> test accuracy: {test_acc}%, test loss: {test_loss}")
        if self.wandb_logs:
            wandb.log({"bit_attack": 0, "test_acc": test_acc})

        data, target = next(iter(calibration_loader))
        bit_iterate = 0
        epsilon = (1 / self.dataset.num_classes)
        acc_threshold = (100 / self.dataset.num_classes) + epsilon
        while test_acc > acc_threshold:
            bit_iterate += 1
            self.attack.progressive_bit_search(model, data, target)
            test_loss, test_acc = evaluate(model, test_loader)
            logging.info(f"bit_attack: {bit_iterate}, test_acc: {test_acc}")
            if self.wandb_logs:
                wandb.log({"bit_attack": bit_iterate, "test_acc": test_acc})
        logging.info(
            f"hamming distance between clean model and attacked model: {hamming_distance_model(self.model, model)}")
        self.attack.save_attack_logs()

    def plot_weight_histogram(self):
        weight_histogram(self.model, save_path=self.save_path, exp_name=self.exp_name,
                         modules_name=None, wandb_=self.wandb_logs)

    def plot_monitor_range(self, wandb_run_path: str = None):
        plot_monitor_range(self, wandb_run_path)
