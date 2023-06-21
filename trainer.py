import torch
import torch.nn.functional as F

import os
import collections
import pandas as pd
import numpy as np

import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from models.models import get_network_class
from algorithms import get_algorithm_class, supervised
from dataloader.dataloader import data_generator
from configs.data_configs import get_dataset_class
from configs.hparams import get_hparams_class
from utils import AverageMeter, to_device, _save_metrics
from utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics


class trainer(object):
    def __init__(self, args):
        self.ssl_method = args.ssl_method
        self.train_mode = args.train_mode
        if "supervised" in self.train_mode:
            self.ssl_method = "supervised"
        self.sleep_model = args.sleep_model
        self.oversample = args.oversample

        # dataset parameters
        self.dataset = args.dataset
        self.fold_id = args.fold_id
        self.data_percentage = args.data_percentage
        self.augmentation = args.augmentation

        self.device = torch.device(args.device)

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.save_dir = f'{args.save_dir}_{self.dataset}'  # To separate the experiments of different datasets
        self.data_path = args.data_path
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # Specify number of hparams
        if self.dataset == "sleep_edf":
            self.hparams = {**self.hparams_class.alg_hparams_edf[self.sleep_model],
                            **self.hparams_class.train_params}
        elif self.dataset == "shhs":
            self.hparams = {**self.hparams_class.alg_hparams_shhs[self.sleep_model],
                            **self.hparams_class.train_params}
        elif self.dataset == "isruc":
            self.hparams = {**self.hparams_class.alg_hparams_isruc[self.sleep_model],
                            **self.hparams_class.train_params}

        self.dataset_configs.num_clsTran_tasks = len(self.augmentation.split("_"))

    def train(self):
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, self.run_description)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)  # save a copy of training files

        self.metrics = {'accuracy': [], 'f1_score': []}

        # fixing random seed
        fix_randomness(int(self.fold_id))

        # Logging
        self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.ssl_method, self.sleep_model,
                                                           self.train_mode, self.exp_log_dir, self.fold_id)

        # Load data
        self.load_data(self.dataset)

        # get algorithm
        backbone_fe = get_network_class(f"{self.sleep_model}_fe")
        backbone_temporal = get_network_class(f"{self.sleep_model}_temporal")
        classifier = get_network_class("classifier")

        if self.train_mode == "ssl":
            algorithm_class = get_algorithm_class(self.ssl_method)
            algorithm = algorithm_class(backbone_fe, backbone_temporal, classifier, self.dataset_configs,
                                        self.hparams, self.device)


        elif "supervised" in self.train_mode:
            backbone_fe = backbone_fe(self.dataset_configs)
            backbone_temporal = backbone_temporal(self.hparams)
            algorithm = supervised(backbone_fe, backbone_temporal, classifier, self.dataset_configs,
                                   self.hparams)


        elif "ft" in self.train_mode:
            backbone_fe = backbone_fe(self.dataset_configs)
            backbone_temporal = backbone_temporal(self.hparams)
            saved_model_dir = os.path.abspath(os.path.join(self.scenario_log_dir, os.pardir))
            saved_model_dir = os.path.join(saved_model_dir, "ssl")

            # load saved models
            chkpoint = torch.load(os.path.join(saved_model_dir, "checkpoint.pt"), map_location=self.device)
            backbone_fe.load_state_dict(chkpoint["fe"])

            algorithm = supervised(backbone_fe, backbone_temporal, classifier, self.dataset_configs,
                                   self.hparams)

        else:
            raise NotImplementedError("Training mode not found: {}".format(self.train_mode))

        algorithm.to(self.device)

        # Average meters
        loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

        self.best_acc = 0
        self.best_f1 = 0

        # training..
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            algorithm.train()

            for step, data in enumerate(self.train_dl):
                data = to_device(data, self.device)

                losses, model = algorithm.update(data)
                for key, val in losses.items():
                    loss_avg_meters[key].update(val, self.hparams["batch_size"])

            if self.train_mode != "ssl":  # Evaluate if not in self-supervised mode.
                self.algorithm = algorithm
                self.evaluate()
                self.calc_results_per_run()

                if self.f1 > self.best_f1:  # save best model based on best f1.
                    self.best_f1 = self.f1
                    self.best_acc = self.acc
                    save_checkpoint(self.home_path, model, self.dataset, self.dataset_configs, self.scenario_log_dir,
                                    self.hparams)
                    self.save_results()
                    _save_metrics(self.pred_labels, self.true_labels, self.scenario_log_dir,
                                  self.home_path,
                                  self.dataset_configs.class_names)

            # logging
            self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in loss_avg_meters.items():
                self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                if self.train_mode != "ssl":
                    self.logger.debug(f'Acc:{self.acc:2.4f} \t F1:{self.f1:2.4f} (best: {self.best_f1:2.4f})')
            self.logger.debug(f'-------------------------------------')

        if self.train_mode == "ssl":
            save_checkpoint(self.home_path, model, self.dataset, self.dataset_configs, self.scenario_log_dir,
                            self.hparams)


        # logging metrics at the last fold:
        if self.fold_id == "4" and self.train_mode != "ssl":  # change "4" if you used different k-fold settings.
            self.calc_overall_results()

    def evaluate(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        temporal_encoder = self.algorithm.temporal_encoder.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        temporal_encoder.eval()
        classifier.eval()

        total_loss_ = []

        self.pred_labels = np.array([])
        self.true_labels = np.array([])

        with torch.no_grad():
            for data in self.test_dl:
                data_samples = to_device(data, self.device)
                data = data_samples['sample_ori'].float()
                labels = data_samples['class_labels'].long()

                # forward pass
                features = feature_extractor(data)
                features = temporal_encoder(features)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss_.append(loss.item())
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.pred_labels = np.append(self.pred_labels, pred.cpu().numpy())
                self.true_labels = np.append(self.true_labels, labels.data.cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()  # average loss

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        if self.train_mode == "ssl":
            hparams_class = get_hparams_class(self.ssl_method)
        else:
            hparams_class = get_hparams_class("supervised")
        return dataset_class(), hparams_class()

    def load_data(self, data_type):
        if self.train_mode == "ssl":  # load full data if you are using Self-supervised learning
            self.data_percentage = "100"
        self.train_dl, self.test_dl = data_generator(self.data_path, data_type, self.fold_id, self.data_percentage,
                                                     self.dataset_configs, self.hparams, self.train_mode,
                                                     self.ssl_method, self.augmentation, self.oversample)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self):
        self.acc, self.f1 = _calc_metrics(self.pred_labels, self.true_labels, self.dataset_configs.class_names)

    def save_results(self):
        run_metrics = {'accuracy': self.best_acc, 'f1_score': self.best_f1}
        df = pd.DataFrame(columns=["acc", "f1"])
        df.loc[0] = [self.acc, self.f1]

        for (key, val) in run_metrics.items(): self.metrics[key].append(val)

        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        df.to_excel(scores_save_path, index=False)
        self.results_df = df

    def calc_overall_results(self):
        exp = self.exp_log_dir

        # for exp in experiments:
        results = pd.DataFrame(columns=["acc", "f1"])

        folds_list = os.listdir(exp)
        folds_list = [i for i in folds_list if "_fold_" in i]
        folds_list = [os.path.join(i, self.train_mode) for i in folds_list]
        folds_list.sort()

        folds_ids = [i.split(os.sep)[:1] for i in folds_list]
        folds_ids = [i[0][1:] for i in folds_ids]

        for idx, fold_id in enumerate(folds_list):
            fold_dir = os.path.join(exp, fold_id)
            scores = pd.read_excel(os.path.join(fold_dir, 'scores.xlsx'))
            scores.insert(0, 'fold', folds_ids[idx])
            results = results.append(scores)

        avg_results = results.mean()
        avg_results = pd.DataFrame(avg_results).transpose()
        avg_results.insert(0, "fold", "mean", True)

        results = results.append(avg_results)

        report_save_path_avg = os.path.join(exp, f"results_{self.train_mode}.xlsx")

        self.logger.debug("######## Overall Results: #########")
        self.logger.debug(f"avg_results: \n{avg_results.mean()}")
        self.logger.debug("###################################")

        self.results_df = results
        results.to_excel(report_save_path_avg)
