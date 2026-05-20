
import warnings
import sys
import os
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import wandb
from pathlib import Path
from torch.backends import cudnn

from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.predictor import Predictor
from gluonts.mx.trainer import Trainer as Trainer_gluonts

from multi_gran_generator import creat_coarse_data_from_graph_different
from pdhd_estimator import pdhdEstimator
from trainer import Trainer
from feature import ZeroTimeFeature, fourier_time_features_from_frequency
from metrics import metric
from mfdfa_toolkit import MFDFA

warnings.filterwarnings("ignore", category=FutureWarning)
np.set_printoptions(threshold=np.inf)

DEFAULT_SEED = 2020
cudnn.benchmark = False
cudnn.deterministic = False


def set_random_seed(seed_value: int) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def parse_lags_seq(lags_str: str):
    return [int(x.strip()) for x in str(lags_str).split(',') if x.strip()]


def print_variable_details(variable):
    var = str(variable)
    var_type = type(variable).__name__

    if len(var) < 100:
        if hasattr(variable, "shape"):
            print("Details:", var, var_type, variable.shape)
        elif hasattr(variable, "__len__"):
            print("Details:", var, var_type, len(variable))
        else:
            print("Details:", var, var_type)
    else:
        if hasattr(variable, "shape"):
            print("Details:", var_type, variable.shape)
        elif hasattr(variable, "__len__"):
            print("Details:", var_type, len(variable))
        else:
            print("Details:", var_type)

    if hasattr(variable, "keys"):
        print(variable.keys())


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='pdhd', help='model name')
    parser.add_argument('--dataset', type=str, default="solar", help='dataset name')
    parser.add_argument('--cuda_num', type=str, default='0', help='cuda number')
    parser.add_argument('--result_path', type=str, default='./results/', help='result path')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-05)
    parser.add_argument('--diff_steps', type=int, default=100, help='diff steps')
    parser.add_argument('--input_size', type=int, default=552,
                        help='legacy arg; auto-computed from lags_seq at runtime')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mg_dict', type=str, default='1_4',
                        help='multi-granularity list, e.g. 1_4_8')
    parser.add_argument('--num_gran', type=int, default=2,
                        help='number of granularities')
    parser.add_argument('--share_ratio_list', type=str, default="1_0.9",
                        help='diffusion step sharing ratios')
    parser.add_argument('--weight_list', type=str, default="0.9_0.1",
                        help='loss weights per granularity')
    parser.add_argument('--run_num', type=str, default="1", help='run index for outputs')
    parser.add_argument('--wandb_space', type=str, default="test", help='wandb project name')
    parser.add_argument('--wandb_key', type=str, default="your wandb key", help='wandb api key')
    parser.add_argument('--log_metrics', type=str2bool, default="False",
                        help='log metrics to wandb during training')
    parser.add_argument('--loss_weight_list', type=str, default="1_0.1",
                        help='loss weight for diffusion and fractal')
    parser.add_argument('--fractal_condition_weight', type=float, default=1)
    parser.add_argument('--diffusion_condition_weight', type=float, default=1)
    parser.add_argument('--beta_end', type=float, default=0.1)
    parser.add_argument('--graph_percentage', type=float, default=0.95)
    parser.add_argument('--dropout_rate_rnn', type=float, default=0.5)
    parser.add_argument('--num_batches_per_epoch', type=int, default=220)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_length', type=int, default=96)
    parser.add_argument('--lags_seq', type=str, default='1,2,4,8,16,32')
    parser.add_argument('--freq', type=str, default='1H')
    parser.add_argument('--use_time_features', type=str2bool, default="False")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--save_ckpt', type=str2bool, default="False")
    parser.add_argument('--skip_train', type=str2bool, default="False",
                        help='skip training and load checkpoint for eval only')
    parser.add_argument('--ckpt_path', type=str, default="",
                        help='checkpoint dir name under checkpoint/ (for skip_train)')
    parser.add_argument('--use_hgnn', type=str2bool, default="True")
    parser.add_argument('--eval_num_samples', type=int, default=100)
    parser.add_argument('--eval_seed', type=int, default=2020)
    parser.add_argument('--maximum_learning_rate', type=float, default=5e-4)
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--loss_warmup_epochs', type=int, default=60)
    return parser.parse_args()


def get_fractality_distribution(data_train, data_test, mg_dict, series_num):
    for train_datadic in data_train:
        target = train_datadic['target']
        mg_list = []
        for mg_i in range(len(mg_dict)):
            series_list = []
            for s_i in range(series_num):
                single_series = target[mg_i, s_i, :]
                single_stats = MFDFA(single_series)
                hqDq = np.stack((single_stats['hq'], single_stats['Dq']), axis=0)
                series_list.append(hqDq)
            mg_list.append(series_list)
        train_datadic.update({'hqDq': np.array(mg_list)})

    for test_datadic in data_test:
        target = test_datadic['target']
        mg_list = []
        for mg_i in range(len(mg_dict)):
            series_list = []
            for s_i in range(series_num):
                single_series = target[mg_i, s_i, :]
                single_stats = MFDFA(single_series)
                hqDq = np.stack((single_stats['hq'], single_stats['Dq']), axis=0)
                series_list.append(hqDq)
            mg_list.append(series_list)
        test_datadic.update({'hqDq': np.array(mg_list)})

    return data_train, data_test


def dic_listtoarray(dicWithList):
    for key in dicWithList.keys():
        if isinstance(dicWithList[key], list):
            dicWithList[key] = np.array(dicWithList[key])
        elif isinstance(dicWithList[key], str):
            dicWithList[key] = pd.to_datetime(dicWithList[key])


args = parse_args()
set_random_seed(args.seed)

result_path = args.result_path
Path(result_path).mkdir(parents=True, exist_ok=True)

epoch = args.epoch
diff_steps = args.diff_steps
num_gran = args.num_gran
batch_size = args.batch_size
mg_dict = [float(i) for i in str(args.mg_dict).split('_')]
share_ratio_list = [float(i) for i in str(args.share_ratio_list).split('_')]
weight_list = [float(i) for i in str(args.weight_list).split('_')]
weights = weight_list
learning_rate = args.learning_rate
loss_weight_list = [float(i) for i in str(args.loss_weight_list).split('_')]
fractal_condition_weight = args.fractal_condition_weight
diffusion_condition_weight = args.diffusion_condition_weight
beta_end = args.beta_end
graph_percentage = args.graph_percentage
dropout_rate_rnn = args.dropout_rate_rnn
num_batches_per_epoch = args.num_batches_per_epoch
num_workers = args.num_workers
context_length = args.context_length
lags_seq = parse_lags_seq(args.lags_seq)
freq = args.freq
use_time_features = args.use_time_features
use_hgnn = args.use_hgnn
eval_num_samples = args.eval_num_samples
eval_seed = args.eval_seed
maximum_learning_rate = args.maximum_learning_rate
clip_gradient = args.clip_gradient if args.clip_gradient > 0 else None
loss_warmup_epochs = args.loss_warmup_epochs
save_ckpt = args.save_ckpt
skip_train = args.skip_train
ckpt_path = args.ckpt_path

print(f"mg_dict:{mg_dict}")
print(f"share_ratio_list:{share_ratio_list}")

if args.log_metrics:
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.wandb_space, save_code=True, config=args,
               settings=wandb.Settings(silent="true"))
print(args)

device = torch.device(f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
PDHD_DATA_ROOT = os.environ.get("PDHD_DATA_ROOT", str(DEFAULT_DATA_ROOT))

start = pd.Timestamp("01-01-1970 00:00:00", freq=freq)

print('******Dataset Info*******')
print("================================================")
print("prepare the dataset")

with open(f"{PDHD_DATA_ROOT}/formatted_data_corr_HCP_train.json", "r") as f:
    data_json = json.load(f)

data_train_mean = []
for i, sample in enumerate(data_json):
    data_train_mean.append({
        'start': start,
        'target': np.array(sample["target"])[:, :],
        'corr': np.array(sample["corr"])[:, :],
        'item_id': i,
    })

series_num = len(data_train_mean[0]['target'])
train_data = ListDataset(data_train_mean, freq=freq, one_dim_target=False)

with open(f"{PDHD_DATA_ROOT}/formatted_data_corr_HCP_test.json", "r") as f:
    data_json = json.load(f)

data_test_mean = []
for i, sample in enumerate(data_json):
    data_test_mean.append({
        'start': start,
        'target': np.array(sample["target"])[:, :],
        'corr': np.array(sample["corr"])[:, :],
        'item_id': i,
    })

test_data = ListDataset(data_test_mean, freq=freq, one_dim_target=False)

print_variable_details(len(test_data))
print_variable_details(len(train_data))

dataset_train = train_data
dataset_test = test_data

output_train_file = f"{PDHD_DATA_ROOT}/processed_d_data_HCP_train_{mg_dict}_{graph_percentage}.json"
output_test_file = f"{PDHD_DATA_ROOT}/processed_d_data_HCP_test_{mg_dict}_{graph_percentage}.json"

if os.path.exists(output_train_file):
    print('*** have read the json files! ***')
    with open(output_train_file, 'r') as json_file:
        data_train = json.load(json_file)
    with open(output_test_file, 'r') as json_file:
        data_test = json.load(json_file)
else:
    print('*** need save the json files! ***')
    data_train, data_test = creat_coarse_data_from_graph_different(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        mg_dict=mg_dict,
        series_num=series_num,
        per=graph_percentage,
    )
    data_train, data_test = get_fractality_distribution(
        data_train, data_test, mg_dict=mg_dict, series_num=series_num
    )

    def dic_arraytolist(dicWithArray):
        for key in dicWithArray.keys():
            if isinstance(dicWithArray[key], np.ndarray):
                dicWithArray[key] = dicWithArray[key].tolist()

    for t_data in data_train:
        dic_arraytolist(t_data)
    for t_data in data_test:
        dic_arraytolist(t_data)

    with open(output_train_file, 'w') as json_file:
        json.dump(data_train, json_file, default=str)
    with open(output_test_file, 'w') as json_file:
        json.dump(data_test, json_file, default=str)

for t_data in data_train:
    dic_listtoarray(t_data)
for t_data in data_test:
    dic_listtoarray(t_data)

prediction_length = 96
time_step = data_train[0]['target'].shape[2]
series_num = data_train[0]['target'].shape[1]
shape = data_train[0]['target'].shape

for i in range(len(data_train)):
    data_train[i]['target'] = data_train[i]['target'].reshape(-1, shape[2])

for i in range(len(data_test)):
    data_test[i]['target'] = data_test[i]['target'].reshape(-1, shape[2])

data_train = ListDataset(data_train, freq=freq, one_dim_target=False)
data_test = ListDataset(data_test, freq=freq, one_dim_target=False)

print("================================================")
print("initlize the estimator")

if use_time_features:
    time_feature_objs = fourier_time_features_from_frequency(freq)
else:
    time_feature_objs = [ZeroTimeFeature()]

input_size = int(series_num * (len(lags_seq) + 1))
print(
    f"context_length={context_length}, lags_seq={lags_seq}, "
    f"auto_input_size={input_size}, time_feature_dims={1 if not use_time_features else len(time_feature_objs)}"
)

num_cells = int(series_num * 0.5)

if args.model_name == 'pdhd':
    estimator = pdhdEstimator(
        target_dim=int(series_num),
        prediction_length=prediction_length,
        context_length=context_length,
        cell_type='GRU',
        input_size=input_size,
        freq=freq,
        loss_type='l2',
        scaling=True,
        diff_steps=diff_steps,
        share_ratio_list=share_ratio_list,
        beta_end=beta_end,
        beta_schedule="linear",
        dropout_rate=dropout_rate_rnn,
        weights=weights,
        num_cells=num_cells,
        num_gran=num_gran,
        seed=args.seed,
        trainer=Trainer(
            device=device,
            epochs=epoch,
            learning_rate=learning_rate,
            num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size,
            maximum_learning_rate=maximum_learning_rate,
            clip_gradient=clip_gradient,
            loss_warmup_epochs=loss_warmup_epochs,
            log_metrics=args.log_metrics,
        ),
        lags_seq=lags_seq,
        time_features=time_feature_objs,
        use_hgnn=use_hgnn,
        loss_weight_list=loss_weight_list,
        fractal_condition_weight=fractal_condition_weight,
        diffusion_condition_weight=diffusion_condition_weight,
    )
elif args.model_name == 'DeepAR':
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        num_cells=num_cells,
        trainer=Trainer_gluonts(
            epochs=epoch,
            num_batches_per_epoch=30,
            batch_size=batch_size,
        ),
    )
else:
    raise ValueError(f"Unsupported model_name: {args.model_name}")

def build_save_path():
    return (
        f"model_{args.model_name}_ds_{args.dataset}_epoch{args.epoch}_"
        f"mg_dict{args.mg_dict}_lr_{args.learning_rate}_prediciton{prediction_length}_"
        f"context_{context_length}_run_{args.run_num}"
    )


Train_indictor = 0 if skip_train else 1
print('Train_indictor', Train_indictor)
sys.stdout.flush()

if Train_indictor == 1:
    print("================================================")
    print("start training the network")
    if args.model_name == 'pdhd':
        predictor = estimator.train(
            data_train, num_workers=num_workers, validation_data=data_test)
    else:
        predictor = estimator.train(
            dataset_train, num_workers=8, validation_data=dataset_test)

    save_path = build_save_path()

    if save_ckpt:
        ckpt_dir = Path(f"checkpoint/{save_path}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        predictor.serialize(ckpt_dir)
        print('Saved!')
    else:
        print('Skip checkpoint save (save_ckpt=False)')

if Train_indictor == 0:
    print('No need to train!')
    sys.stdout.flush()
    load_path = ckpt_path if ckpt_path else build_save_path()
    print(f"load checkpoint: checkpoint/{load_path}/", flush=True)
    predictor = Predictor.deserialize(Path(f"checkpoint/{load_path}/"))
    sys.stdout.flush()
    print('Loaded!', flush=True)

num_samples = eval_num_samples
print(f"eval_num_samples={num_samples}")
if eval_seed >= 0:
    print(f"eval_seed={eval_seed}")
    set_random_seed(eval_seed)

print("===============================================", flush=True)
print("make predictions", flush=True)
forecast_it, ts_it = make_evaluation_predictions(
    dataset=data_test,
    predictor=predictor,
    num_samples=num_samples,
)
forecasts = list(forecast_it)
targets = list(ts_it)

targets_list = []
forecasts_list = []
target_dim = estimator.target_dim

for cur_gran_index, _cur_gran in enumerate(mg_dict):
    targets_cur = []
    predict_cur = copy.deepcopy(forecasts)

    for i in range(len(targets)):
        targets_cur.append(
            targets[i].iloc[:, (cur_gran_index * target_dim):((cur_gran_index + 1) * target_dim)])
        targets_cur[i] = np.array(targets_cur[i])[:, :]

    for day in range(len(forecasts)):
        predict_cur[day] = forecasts[day].samples[
            :, :, (cur_gran_index * target_dim):((cur_gran_index + 1) * target_dim)
        ]

    targets_list.append(np.array(targets_cur))
    forecasts_list.append(predict_cur)

full_forecasts_list = np.array(targets_list[0])

warnings.filterwarnings("ignore")
dataset_name = 'HCP'

for cur_gran_index, cur_gran in enumerate(mg_dict):
    if cur_gran_index != 0:
        continue

    forecasts_all = np.array(forecasts_list)[cur_gran_index]
    targets_gran_list = np.array(targets_list[cur_gran_index])[:, -prediction_length:, :]
    forecasts_mean = forecasts_all.mean(axis=1)
    print(forecasts_mean.shape)

    mae, _, rmse, mape, _, _, _, nd, _ = metric(
        forecasts_mean, targets_gran_list
    )

    sample_mae_list = []
    for sample_i in range(num_samples):
        sample_mae_list.append(
            metric(forecasts_all[:, sample_i], targets_gran_list)[0]
        )
    sample_mae_mean = float(np.mean(sample_mae_list))
    sample_mae_std = float(np.std(sample_mae_list))

    full_forecasts_list[:, -prediction_length:, :] = forecasts_mean

    print(f"=======Evaluation results for {cur_gran} h samples")
    print('mae:{}, rmse:{}, mape:{},nd:{}'.format(mae, rmse, mape, nd))
    print(
        'sample_mae_mean:{}, sample_mae_std:{}'.format(sample_mae_mean, sample_mae_std)
    )

    array_as_list = full_forecasts_list.tolist()
    prediction_data_dir = os.path.join(result_path, "prediction_data")
    os.makedirs(prediction_data_dir, exist_ok=True)
    prediction_data_path = os.path.join(
        prediction_data_dir,
        f"prediction_data_HCP_test_{prediction_length}_{context_length}_run_{args.run_num}.json",
    )
    with open(prediction_data_path, 'w') as json_file:
        json.dump(array_as_list, json_file)
    print("saved", prediction_data_path)

    filename = f"{result_path}/output_{dataset_name}_{args.model_name}_{mg_dict}h_{cur_gran}h_{weights}_ratio{share_ratio_list}.csv"
    config_str1 = f"{epoch}, {diff_steps}, {learning_rate}, {batch_size}, {num_cells}, {prediction_length}\n"
    config_str2 = f"{loss_weight_list}, {fractal_condition_weight}, {diffusion_condition_weight}\n"
    result_str = f"{mae}, {rmse}, {mape}, {nd}\n"
    note = "Note:only fractal and rnn condition, with bn \n"

    with open(filename, mode="a") as f:
        f.write(note)
        f.write("epoch,diff_steps,learning_rate,batch_size,num_cells,prediction_length\n")
        f.write(config_str1)
        f.write("loss_weight_list,fractal_condition_weight,diffusion_condition_weight\n")
        f.write(config_str2)
        f.write("MAE,RMSE,MAPE,nd\n")
        f.write(result_str)
        f.write('--------------------------------------------------------------------------\n')

        if args.log_metrics:
            wandb.log({
                f'mae_{cur_gran}': mae,
                f'rmse_{cur_gran}': rmse,
                f'mape_{cur_gran}': mape,
                f'nd_{cur_gran}': nd,
            })
