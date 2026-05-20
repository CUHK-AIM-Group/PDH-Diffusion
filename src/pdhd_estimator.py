# use multiple datasets and multiple networks
from typing import List, Optional, NamedTuple
from functools import partial

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader

from gluonts.env import env
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.model.estimator import Estimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.model.predictor import Predictor
from gluonts.transform import SelectFields, Transformation
from gluonts.itertools import maybe_len
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
import numpy as np
import copy
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from trainer import Trainer
from feature import fourier_time_features_from_frequency, lags_for_fourier_time_features_from_frequency
from estimator import PyTorchEstimator
from utils import get_module_forward_input_names
from dataset import TransformedIterableDataset
# from pdhd_network import pdhdPredictionNetwork, pdhdTrainingNetwork
from pdhd_network_Mamba import pdhdPredictionNetwork, pdhdTrainingNetwork

# this class inherit the PyTorchEstimator class, and overwrite
# the functions in the previous class

def print_details(variable):
    var_name = [k for k, v in locals().items() if v is variable]
    print(var_name[0],variable)

class TrainOutput(NamedTuple):  # added
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor


class pdhdEstimator(PyTorchEstimator):
    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        num_gran: int,
        trainer: Optional[Trainer] = None,
        context_length: Optional[int] = None,
        num_layers: int = 2,
        num_cells: int = 40,
        cell_type: str = "LSTM",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [1],
        embedding_dimension: int = 5,
        conditioning_length: int = 100,
        loss_type: str = "l2",
        # diff_steps: int = 100,
        # beta_end=0.1,
        diff_steps: Optional[ List[int] ]= None,
        beta_end:  Optional[List[float]]=None,
        beta_schedule="linear",
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        scaling: bool = True,
        share_ratio_list: Optional[List[float]] = [1],
        weights: Optional[List[float]] = None,
        pick_incomplete: bool = False,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        loss_weight_list: List[float]=[1,0.1],
        fractal_condition_weight:float=1,
        diffusion_condition_weight:float=1,
        use_hgnn: bool = True,
        **kwargs,
    ) -> None:
        if trainer is None:
            trainer = Trainer()
        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq
        self.context_length = (
            context_length if context_length is not None else prediction_length
        )  # context_length default = prediction_length

        self.input_size = input_size
        # print_details(input_size) #1481
        self.prediction_length = prediction_length
        # print_details(prediction_length) #24
        self.target_dim = target_dim

        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type  # the type of RNN cell LSTM
        self.num_parallel_samples = num_parallel_samples  # parallel samples 100
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        # print_details(cardinality) #[1]
        self.embedding_dimension = embedding_dimension
        self.num_gran = num_gran

        self.conditioning_length = conditioning_length

        # Backward-compatible normalization:
        # run scripts often pass scalar diff_steps/beta_end, while current
        # training network expects per-granularity lists.
        if isinstance(diff_steps, list):
            self.diff_steps = diff_steps
        elif diff_steps is None:
            self.diff_steps = [100] * self.num_gran
        else:
            self.diff_steps = [int(diff_steps)] * self.num_gran

        self.loss_type = loss_type  # L1 loss or L2 loss
        if isinstance(beta_end, list):
            self.beta_end = beta_end
        elif beta_end is None:
            self.beta_end = [0.1] * self.num_gran
        else:
            self.beta_end = [float(beta_end)] * self.num_gran

        if len(self.diff_steps) != self.num_gran:
            raise ValueError(
                f"diff_steps length {len(self.diff_steps)} != num_gran {self.num_gran}"
            )
        if len(self.beta_end) != self.num_gran:
            raise ValueError(
                f"beta_end length {len(self.beta_end)} != num_gran {self.num_gran}"
            )

        self.beta_schedule = beta_schedule  # linear or cosine etc.
        self.residual_layers = residual_layers  # 8
        self.residual_channels = residual_channels  # 8
        self.dilation_cycle_length = dilation_cycle_length  # 2

        self.loss_weight_list=loss_weight_list
        self.fractal_condition_weight=fractal_condition_weight
        self.diffusion_condition_weight=diffusion_condition_weight
        self.use_hgnn = use_hgnn

        self.weights = (
            weights
            if weights is not None
            else [0.8, 0.2]  # default weight of the loss
        )

        self.share_ratio_list = share_ratio_list  # add another argument
        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq)
        )

        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )

        # context_length + max(lags_seq) 24 + 168 = 192
        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = pick_incomplete
        self.scaling = scaling  # whether to scale the input data

        self.train_sampler = ExpectedNumInstanceSampler(
            # num_instances=1.0,
            num_instances=1000.0,# 128.0 1000
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

    def create_transformation(self) -> Transformation:
        """时序数据的转换

        Returns:
            Transformation: 转换链
        """
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),  # expand the target to (1, T)
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),  # add observed values indicator
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),  # add time features
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),  # vstack time features
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0]),  # set static cat
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),  # target dim indicator
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT,
                             expected_ndim=1),  # static cat
            ]
        )



    # 相当于采样，数据集是多个序列，训练样本是每次从一个序列采指定数量的窗口数据作为样本
    # is_pad_field：表示是否进行pad操作
    # ExpectedNumInstanceSampler：采样器，num_instances表示每个序列采样窗口数
    # time_series_fields：表示dynamic特征，采样器会同时采这些特征的窗口数据



    def create_instance_splitter(self, mode: str) -> InstanceSplitter:
        """创建实例分割器

        Args:
            mode (str): 模式

        Returns:
            InstanceSplitter: 实例分割器
        """
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        # print(dir(FieldName))


        return InstanceSplitter(
            # item_id_field=FieldName.ITEM_ID,

            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def create_training_network(self, device: torch.device) -> pdhdTrainingNetwork:
        print(f"self_share_ratio of estimator:{self.share_ratio_list}")
        return pdhdTrainingNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            num_gran=self.num_gran,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            weights=self.weights,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            share_ratio_list=self.share_ratio_list,
            conditioning_length=self.conditioning_length,
            num_parallel_samples = self.num_parallel_samples,
            loss_weight_list=self.loss_weight_list,
            fractal_condition_weight=self.fractal_condition_weight,
            diffusion_condition_weight=self.diffusion_condition_weight,
            use_hgnn=self.use_hgnn,
            # fractal_indicator=
        ).to(device)

    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: pdhdTrainingNetwork,
        device: torch.device,
    ) -> Predictor:

        prediction_network = pdhdPredictionNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_gran=self.num_gran,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            diff_steps=self.diff_steps,
            loss_type=self.loss_type,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            residual_layers=self.residual_layers,
            residual_channels=self.residual_channels,
            dilation_cycle_length=self.dilation_cycle_length,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            share_ratio_list=self.share_ratio_list,
            conditioning_length=self.conditioning_length,
            num_parallel_samples=self.num_parallel_samples,
            use_hgnn=self.use_hgnn,
        ).to(device)

        copy_parameters(trained_network, prediction_network)  # copy parameters
        input_names = get_module_forward_input_names(
            prediction_network)
        prediction_splitter = self.create_instance_splitter(
            "test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )

    def get_metric(self, transformation, trained_net, device, data_test, prefix=""):
        predictor = self.create_predictor(
            transformation=transformation,
            trained_network=trained_net,
            device=device,
        )
        forecast_it, ts_it = make_evaluation_predictions(dataset=data_test,
                                                         predictor=predictor,
                                                         num_samples=100)
        forecasts = list(forecast_it)
        targets = list(ts_it)
        targets_fine = []
        for i in range(7):
            targets_fine.append(targets[i].iloc[:, :137])

        print("=================================================")
        print("make evaluations")

        forecast_fine = copy.deepcopy(forecasts)
        for day in range(7):
            forecast_fine[day].samples = forecasts[day].samples[:, :, :137]
        import warnings
        # Ignore all warnings
        warnings.filterwarnings("ignore")
        evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:],
                                          target_agg_funcs={'sum': np.sum})
        agg_metric, item_metrics = evaluator(
            targets_fine, forecast_fine, num_series=len(data_test))

        print("CRPS:", agg_metric["mean_wQuantileLoss"])
        print("ND:", agg_metric["ND"])
        print("NRMSE:", agg_metric["NRMSE"])
        print("")
        print("CRPS-Sum:", agg_metric["m_sum_mean_wQuantileLoss"])
        print("ND-Sum:", agg_metric["m_sum_ND"])
        print("NRMSE-Sum:", agg_metric["m_sum_NRMSE"])

        CRPS = agg_metric["mean_wQuantileLoss"]
        ND = agg_metric["ND"]
        NRMSE = agg_metric["NRMSE"]
        CRPS_Sum = agg_metric["m_sum_mean_wQuantileLoss"]
        ND_Sum = agg_metric["m_sum_ND"]
        NRMSE_Sum = agg_metric["m_sum_NRMSE"]
        return {f'{prefix}CRPS': CRPS, f'{prefix}ND': ND, f'{prefix}NRMSE': NRMSE, f'{prefix}CRPS_Sum': CRPS_Sum, f'{prefix}ND_Sum': ND_Sum, f'{prefix}NRMSE_Sum': NRMSE_Sum}
