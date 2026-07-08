import logging
import time
import numpy as np

import torch
from torch import nn

from utils.data_utils import (
    get_named_data,
    get_all_bn_params,
    check_device
)

from trainers.averager import Averager
from utils.set import *
from utils.log_info import *
from utils.tool import *


class NormalTrainer(object):

    def __init__(
            self,
            model,
            device,
            criterion,
            optimizer,
            lr_scheduler,
            args,
            **kwargs):


        if kwargs["role"] == "server":

            if "server_index" in kwargs:
                self.server_index = kwargs["server_index"]
            else:
                self.server_index = args.server_index

            self.client_index = None
            self.index = self.server_index


        elif kwargs["role"] == "client":

            if "client_index" in kwargs:
                self.client_index = kwargs["client_index"]
            else:
                self.client_index = args.client_index

            self.server_index = None
            self.index = self.client_index


        else:

            raise NotImplementedError



        self.role = kwargs["role"]

        self.args = args

        self.model = model

        self.device = device

        self.criterion = criterion.to(device)

        self.optimizer = optimizer


        self.param_groups = self.optimizer.param_groups


        self.named_parameters = list(
            self.model.named_parameters()
        )


        if len(self.named_parameters) > 0:

            self._parameter_names = {
                v: k
                for k, v in sorted(
                    self.named_parameters
                )
            }

        else:

            self._parameter_names = {
                v: f"noname.{i}"
                for param_group in self.param_groups
                for i, v in enumerate(
                    param_group["params"]
                )
            }



        self.averager = Averager(
            self.args,
            self.model
        )


        self.lr_scheduler = lr_scheduler



    # ======================================================
    # Basic interface
    # ======================================================


    def epoch_init(self):
        pass


    def epoch_end(self):
        pass


    def track(
            self,
            tracker,
            summary_n_samples,
            model,
            loss,
            end_of_epoch,
            checkpoint_extra_name="centralized",
            things_to_track=[]):

        pass



    def update_state(self, **kwargs):

        self.update_loss_state(
            **kwargs
        )



    def get_model_named_modules(self):

        return dict(
            self.model.cpu().named_modules()
        )



    def get_model(self):

        return self.model



    def get_model_params(self):

        return self.model.cpu().state_dict()



    def set_model_params(
            self,
            model_parameters):

        self.model.load_state_dict(
            model_parameters
        )



    # ======================================================
    # BN / feature utilities
    # ======================================================


    def set_feature_align_means(
            self,
            feature_align_means):

        self.feature_align_means = feature_align_means

        self.align_feature_loss.feature_align_means = (
            feature_align_means
        )



    def get_feature_align_means(self):

        return self.feature_align_means



    def get_model_bn(self):

        return get_all_bn_params(
            self.model
        )



    def set_model_bn(
            self,
            all_bn_params):


        for module_name, module in self.model.named_modules():


            if type(module) is nn.BatchNorm2d:


                module.weight.data = (
                    all_bn_params[
                        module_name + ".weight"
                    ]
                )


                module.bias.data = (
                    all_bn_params[
                        module_name + ".bias"
                    ]
                )


                module.running_mean = (
                    all_bn_params[
                        module_name + ".running_mean"
                    ]
                )


                module.running_var = (
                    all_bn_params[
                        module_name + ".running_var"
                    ]
                )


                module.num_batches_tracked = (
                    all_bn_params[
                        module_name + ".num_batches_tracked"
                    ]
                )



    # ======================================================
    # Gradient utilities
    # ======================================================


    def get_model_grads(self):

        return get_named_data(
            self.model,
            mode="GRAD",
            use_cuda=True
        )



    def set_grad_params(
            self,
            named_grads):

        self.model.train()

        self.optimizer.zero_grad()


        for name, parameter in self.model.named_parameters():

            if name in named_grads:

                parameter.grad.copy_(
                    named_grads[name]
                    .data
                    .to(self.device)
                )



    def clear_grad_params(self):

        self.optimizer.zero_grad()



    def update_model_with_grad(self):

        self.model.to(
            self.device
        )

        self.optimizer.step()



    def get_optim_state(self):

        return self.optimizer.state



    def clear_optim_buffer(self):

        for group in self.optimizer.param_groups:

            for p in group["params"]:

                param_state = (
                    self.optimizer.state[p]
                )

                if "momentum_buffer" in param_state:

                    param_state[
                        "momentum_buffer"
                    ].zero_()



    # ======================================================
    # LR schedule
    # ======================================================


    def lr_schedule(
            self,
            progress):


        if self.lr_scheduler is not None:

            self.lr_scheduler.step(
                progress
            )

        else:

            logging.debug(
                "No lr scheduler"
            )



    def warmup_lr_schedule(
            self,
            iterations):


        if self.lr_scheduler is not None:

            self.lr_scheduler.warmup_step(
                iterations
            )



    # ======================================================
    # Data iterator
    # ======================================================


    def get_train_batch_data(
            self,
            train_dataloader):


        try:

            train_batch_data = next(
                self.train_local_iter
            )


        except:

            self.train_local_iter = iter(
                train_dataloader
            )

            train_batch_data = next(
                self.train_local_iter
            )


        return train_batch_data

# trainers/normal_trainer.py  Part 2/2


    # ======================================================
    # Main local training
    # Support:
    #   local + proxy1 + proxy2
    #   local only
    # ======================================================

    def train_mix_dataloader(
            self,
            epoch,
            trainloader,
            device,
            **kwargs):


        train_start_time = time.time()


        self.model.to(device)

        self.model.train()

        self.model.training = True



        loss_avg = AverageMeter()

        acc = AverageMeter()



        logging.debug(
            f"{self.role}_{self.index}: "
            f"start epoch {epoch}"
        )



        for batch_idx, batch in enumerate(trainloader):


            # ==================================================
            # Flexible dataloader parsing
            # ==================================================

            if len(batch) == 6:

                x1, x2, x3, y1, y2, y3 = batch


                xs = [
                    x1,
                    x2,
                    x3
                ]


                ys = [
                    y1,
                    y2,
                    y3
                ]



            elif len(batch) == 4:


                x1, x2, y1, y2 = batch


                xs = [
                    x1,
                    x2
                ]


                ys = [
                    y1,
                    y2
                ]



            elif len(batch) == 2:


                x1, y1 = batch


                xs = [
                    x1
                ]


                ys = [
                    y1
                ]


            else:

                raise RuntimeError(
                    f"Unsupported batch format: "
                    f"{len(batch)}"
                )



            xs = [
                x.to(device)
                for x in xs
                if x is not None
            ]


            ys = [
                y.to(device)
                for y in ys
                if y is not None
            ]



            if len(xs) == 0:

                continue



            x = torch.cat(
                xs,
                dim=0
            )


            y = torch.cat(
                ys,
                dim=0
            )


            y = y.long()



            batch_size = x.size(0)



            self.optimizer.zero_grad()



            output = self.model(x)



            loss = self.criterion(
                output,
                y
            )



            # ==================================================
            # FedProx
            # ==================================================

            if getattr(
                    self.args,
                    "fedprox",
                    False):


                previous_model = kwargs.get(
                    "previous_model",
                    None
                )


                if previous_model is not None:


                    fed_prox_reg = 0.0


                    for name, param in self.model.named_parameters():


                        if name in previous_model:


                            fed_prox_reg += (

                                torch.norm(
                                    param
                                    -
                                    previous_model[name]
                                    .to(device)
                                ) ** 2

                            )



                    loss += (

                        self.args.fedprox_mu
                        /
                        2.0

                    ) * fed_prox_reg



            loss.backward()



            self.optimizer.step()



            # ==================================================
            # SCAFFOLD
            # ==================================================

            if getattr(
                    self.args,
                    "scaffold",
                    False):


                c_model_global = kwargs.get(
                    "c_model_global",
                    None
                )


                c_model_local = kwargs.get(
                    "c_model_local",
                    None
                )



                if (
                    c_model_global is not None
                    and
                    c_model_local is not None
                ):


                    for name, param in self.model.named_parameters():


                        if (
                            name in c_model_global
                            and
                            name in c_model_local
                        ):


                            correction = (

                                c_model_global[name]
                                -
                                c_model_local[name]

                            )


                            correction = (
                                correction
                                .to(param.device)
                            )


                            param.data -= (
                                0.000001
                                *
                                correction
                            )



            prec1, prec5, correct, pred, _ = accuracy(
                output.data,
                y.data,
                topk=(1, 5)
            )



            loss_avg.update(
                loss.item(),
                batch_size
            )


            acc.update(
                prec1.item(),
                batch_size
            )



            n_iter = (
                epoch * len(trainloader)
                +
                batch_idx
            )



            log_info(
                'scalar',
                f'{self.role}_{self.index}_train_loss',
                loss_avg.avg,
                step=n_iter,
                record_tool=self.args.record_tool,
                wandb_record=self.args.wandb_record
            )


            log_info(
                'scalar',
                f'{self.role}_{self.index}_train_acc',
                acc.avg,
                step=n_iter,
                record_tool=self.args.record_tool,
                wandb_record=self.args.wandb_record
            )



        train_time_sec = (
            time.time()
            -
            train_start_time
        )



        self.last_train_stats = {

            "epoch":
                int(epoch),

            "train_loss":
                float(loss_avg.avg),

            "train_acc":
                float(acc.avg),

            "num_batches":
                int(len(trainloader)),

            "train_time_sec":
                float(train_time_sec),

        }



        if torch.cuda.is_available():

            self.last_train_stats[
                "cuda_max_memory_mb"
            ] = float(
                torch.cuda.max_memory_allocated()
                /
                1024
                /
                1024
            )


        else:

            self.last_train_stats[
                "cuda_max_memory_mb"
            ] = 0.0




    # ======================================================
    # Server evaluation
    # ======================================================


    def test_on_server_for_round(
            self,
            round,
            testloader,
            device):


        self.model.to(device)

        self.model.eval()



        test_acc_avg = AverageMeter()

        test_loss_avg = AverageMeter()



        total_loss_avg = 0

        total_acc_avg = 0



        with torch.no_grad():


            for batch_idx, (x, y) in enumerate(testloader):


                x = x.to(device)

                y = y.to(device).view(-1)



                batch_size = x.size(0)



                output = self.model(x)



                loss = self.criterion(
                    output,
                    y.long()
                )



                prec1, _ = accuracy(
                    output.data,
                    y
                )



                test_acc_avg.update(
                    prec1.item(),
                    batch_size
                )


                test_loss_avg.update(
                    loss.item(),
                    batch_size
                )



                n_iter = (
                    (round - 1)
                    *
                    len(testloader)
                    +
                    batch_idx
                )



                log_info(
                    'scalar',
                    f'{self.role}_{self.index}_test_acc',
                    test_acc_avg.avg,
                    step=n_iter,
                    record_tool=self.args.record_tool,
                    wandb_record=self.args.wandb_record
                )



                total_loss_avg += test_loss_avg.avg

                total_acc_avg += test_acc_avg.avg



            if len(testloader) > 0:

                total_acc_avg /= len(testloader)

                total_loss_avg /= len(testloader)



            log_info(
                'scalar',
                f'{self.role}_{self.index}_total_acc',
                total_acc_avg,
                step=round,
                record_tool=self.args.record_tool,
                wandb_record=self.args.wandb_record
            )


            log_info(
                'scalar',
                f'{self.role}_{self.index}_total_loss',
                total_loss_avg,
                step=round,
                record_tool=self.args.record_tool,
                wandb_record=self.args.wandb_record
            )



            return total_acc_avg



    # ======================================================
    # Client test interface
    # ======================================================


    def test(
            self,
            epoch,
            testloader,
            device):


        return self.test_on_server_for_round(
            epoch,
            testloader,
            device
        )