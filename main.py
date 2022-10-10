import os
import argparse
import warnings
from trainer import trainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()

# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='test',   type=str, help='experiment name')
parser.add_argument('--run_description',        default='run_description',     type=str, help='run name')

# ========= Select methods ============
parser.add_argument('--sleep_model',              default='dsn',          type=str, help='dsn, attnsleep, cnn1d')
parser.add_argument('--ssl_method',              default='simclr',        type=str, help='simclr, cpc, ts_tcc, clsTran')
parser.add_argument('--train_mode',              default='ssl',            type=str, help='supervised, ssl, (ft)fine_tune, (lc)linear_classifier')
parser.add_argument('--oversample',              default=False,           type=bool, help='apply oversampling or not?')

# ========= Select the DATASET ==============
parser.add_argument('--dataset',                default='sleep_edf',           type=str, help='sleep_edf, shhs')
parser.add_argument('--fold_id',                default='0',               type=str, help='(0,1,2,3,4) for 5-fold CV')
parser.add_argument('--data_percentage',        default='100',             type=str, help='1,5,10,100')
parser.add_argument('--augmentation',           default='noise_permute',   type=str, help='augmentation type for simclr')


# ========= Experiment settings ===============
parser.add_argument('--data_path',              default=r'data',           type=str,   help='Path containing dataset')
parser.add_argument('--num_runs',               default=1,                 type=int,   help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0',            type=str,   help='cpu or cuda')


args = parser.parse_args()

if __name__ == "__main__":
    trainer = trainer(args)
    trainer.train()
