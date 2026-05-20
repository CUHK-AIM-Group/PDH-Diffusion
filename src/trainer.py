import time
from typing import Optional, Union

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from gluonts.core.component import validated
from gluonts.transform import Transformation
from gluonts.dataset.common import Dataset
import wandb

from data_provider.data_factory import data_provider


class Trainer:
    @validated()
    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        maximum_learning_rate: Optional[float] = None,
        clip_gradient: Optional[float] = None,
        loss_warmup_epochs: int = 30,
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_batches_per_epoch = num_batches_per_epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if maximum_learning_rate is None or float(maximum_learning_rate) <= 0:
            self.maximum_learning_rate = self.learning_rate
        else:
            self.maximum_learning_rate = max(
                self.learning_rate, float(maximum_learning_rate)
            )
        self.clip_gradient = clip_gradient
        self.loss_warmup_epochs = max(1, int(loss_warmup_epochs))
        self.device = device
        self.total_step = 0
        self.log_metrics = kwargs.get('log_metrics')
        print(f'self.log_metrics: {self.log_metrics}')
        print(
            f'effective_lr={self.learning_rate}, effective_max_lr={self.maximum_learning_rate}, '
            f'clip_gradient={self.clip_gradient}, loss_warmup_epochs={self.loss_warmup_epochs}'
        )

    def _get_data(self, flag, data):
        data_set, data_loader = data_provider(self.args, flag, data)
        return data_set, data_loader

    def __call__(
        self,
        net: nn.Module,
        train_iter: DataLoader,
        validation_iter: Optional[DataLoader] = None,
        validation_dataset: Optional[Dataset] = None,
        transformation: Transformation = None,
        estimator=None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        optimizer = Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        lr_scheduler = None
        if self.maximum_learning_rate > self.learning_rate:
            lr_scheduler = OneCycleLR(
                optimizer,
                max_lr=self.maximum_learning_rate,
                steps_per_epoch=self.num_batches_per_epoch,
                epochs=self.epochs,
            )
        avg_last_epoch_loss=1e8
        for epoch_no in range(self.epochs):
            print('total epoch',self.epochs)
            self.total_step += 1

            net.set_epoch(epoch_no)

            if self.log_metrics == True:
                wandb.log({"train/epoch": epoch_no}, step=self.total_step)
            tic = time.time()
            
            cumm_epoch_loss = 0.0
            cumm_loss_frac=0.0
            cumm_loss_diffusion=0.0
            total = self.num_batches_per_epoch - 1

            net.train()
            # Smoothly anneal from output[0] to output[1] to avoid a hard phase switch.
            loss_blend_lambda = min(1.0, float(epoch_no) / float(self.loss_warmup_epochs))
        
            with tqdm(train_iter, total=total) as it:
                
                for batch_no, data_entry in enumerate(it, start=1):
                    optimizer.zero_grad()
                    inputs = [v.to(self.device) for v in data_entry.values()]
                    output = net(*inputs)
                    mt_stats = getattr(net, "last_mt_stats", None)

                    if isinstance(output, (list, tuple)):
                        loss = (1.0 - loss_blend_lambda) * output[0] + loss_blend_lambda * output[1]
                    else:
                        loss = output
                    loss_frac=output[-2]
                    loss_diffusion=output[-1]



                    cumm_loss_frac+=loss_frac.item()
                    cumm_loss_diffusion+=loss_diffusion.item()

                    cumm_epoch_loss += loss.item()
                    avg_epoch_loss = cumm_epoch_loss / batch_no
                    if batch_no%50==0:
                        postfix = {
                            "epoch": f"{epoch_no + 1}/{self.epochs}",
                            "avg_loss": avg_epoch_loss,
                            "cumm_loss_frac": cumm_loss_frac / batch_no,
                            "cumm_loss_diffusion": cumm_loss_diffusion / batch_no,
                        }
                        if isinstance(mt_stats, dict) and mt_stats:
                            postfix.update(
                                {
                                    "mt_lambda": loss_blend_lambda,
                                    "mt_raw_total": mt_stats.get("raw_total_loss", 0.0),
                                    "mt_norm_f": mt_stats.get("norm_loss_frac", 0.0),
                                    "mt_norm_d": mt_stats.get("norm_loss_diff", 0.0),
                                }
                            )
                        it.set_postfix(postfix, refresh=False)
                        if self.log_metrics == True:
                            wandb.log({'train/loss': avg_epoch_loss},
                                    step=self.total_step)
                            wandb.log({'train/loss_frac': cumm_loss_frac/ batch_no},
                                    step=self.total_step)
                            wandb.log({'train/loss_diffusion': cumm_loss_diffusion/ batch_no},
                                    step=self.total_step)
                    loss.backward()
                    if self.clip_gradient is not None:
                        nn.utils.clip_grad_norm_(
                            net.parameters(), self.clip_gradient)

                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    if self.num_batches_per_epoch == batch_no:
                        break
                it.close()

            print('avg_last_epoch_loss',avg_last_epoch_loss)
            print('cumm_epoch_loss',cumm_epoch_loss)
            
            avg_last_epoch_loss=cumm_epoch_loss



            vali_loss_best = 1e8

            if validation_iter is not None:
                net.eval()
                cumm_epoch_loss_val = 0.0
                cumm_loss_frac=0.0
                cumm_loss_diffusion=0.0

                with tqdm(validation_iter, total=total, colour="green") as it:

                    for batch_no, data_entry in enumerate(it, start=1):
                        inputs = [v.to(self.device)
                                  for v in data_entry.values()]
                        with torch.no_grad():
                            output = net(*inputs)
                            mt_stats = getattr(net, "last_mt_stats", None)

                        if isinstance(output, (list, tuple)):
                            loss = (1.0 - loss_blend_lambda) * output[0] + loss_blend_lambda * output[1]
                        else:
                            loss = output

                        loss_frac=output[-2]
                        loss_diffusion=output[-1]

                        cumm_epoch_loss_val += loss.item()
                        cumm_loss_frac+=loss_frac.item()
                        cumm_loss_diffusion+=loss_diffusion.item()

                        avg_epoch_loss_val = cumm_epoch_loss_val / batch_no
                        if batch_no%10==0:
                            postfix = {
                                "epoch": f"{epoch_no + 1}/{self.epochs}",
                                "avg_loss": avg_epoch_loss,
                                "avg_val_loss": avg_epoch_loss_val,
                                "loss_frac": cumm_loss_frac / batch_no,
                                "loss_diffusion": cumm_loss_diffusion / batch_no,
                            }
                            if isinstance(mt_stats, dict) and mt_stats:
                                postfix.update(
                                    {
                                        "mt_lambda": mt_stats.get("lambda_t", 0.0),
                                        "mt_raw_total": mt_stats.get("raw_total_loss", 0.0),
                                        "mt_norm_f": mt_stats.get("norm_loss_frac", 0.0),
                                        "mt_norm_d": mt_stats.get("norm_loss_diff", 0.0),
                                    }
                                )
                            it.set_postfix(postfix, refresh=False)
                            if self.log_metrics == True:
                                wandb.log({'val/loss': avg_epoch_loss_val},
                                        step=self.total_step)
                                wandb.log({'val/loss_frac': cumm_loss_frac/ batch_no},
                                    step=self.total_step)
                                wandb.log({'val/loss_diffusion': cumm_loss_diffusion/ batch_no},
                                    step=self.total_step)
                        if self.num_batches_per_epoch == batch_no:
                            break

                it.close()
               
            # Log elapsed time per epoch for runtime monitoring.
            toc = time.time()
