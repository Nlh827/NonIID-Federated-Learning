import logging
import copy
import time

from algorithms_standalone.basePS.client import Client
from model.build import create_model

from utils.fedselect_utils import (
    select_global_params_for_upload,
)


class FedAVGClient(Client):

    def __init__(
            self,
            client_index,
            train_ori_data,
            train_ori_targets,
            test_dataloader,
            train_data_num,
            test_data_num,
            train_cls_counts_dict,
            device,
            args,
            model_trainer,
            vae_model,
            dataset_num):


        super().__init__(
            client_index,
            train_ori_data,
            train_ori_targets,
            test_dataloader,
            train_data_num,
            test_data_num,
            train_cls_counts_dict,
            device,
            args,
            model_trainer,
            vae_model,
            dataset_num
        )


        self.global_epochs_per_round = (
            self.args.global_epochs_per_round
        )


        # =========================
        # SCAFFOLD
        # =========================

        if self.args.scaffold:

            self.c_model_local = create_model(
                self.args,
                model_name=self.args.model,
                output_dim=self.args.model_output_dim
            )


            for _, params in self.c_model_local.named_parameters():

                params.data = params.data * 0



    # ======================================================
    # Learning rate schedule
    # ======================================================

    def lr_schedule(
            self,
            num_iterations,
            warmup_epochs):


        round_idx = (
            self.client_timer.local_comm_round_idx
        )


        if self.args.sched == "no":

            return


        if round_idx < warmup_epochs:

            self.trainer.warmup_lr_schedule(
                round_idx * num_iterations
            )

        else:

            self.trainer.lr_schedule(
                round_idx
            )



    # ======================================================
    # Test
    # ======================================================

    def test(self, epoch):

        acc_avg = self.trainer.test(
            epoch,
            self.test_dataloader,
            self.device
        )

        return acc_avg



    # ======================================================
    # FedAvg local training
    # ======================================================

    def fedavg_train(
            self,
            share_data1,
            share_data2,
            share_y,
            round_idx=None,
            global_other_params=None,
            shared_params_for_simulation=None,
            **kwargs):


        train_start_time = time.time()


        use_fedselect = getattr(
            self.args,
            "fedselect",
            False
        )


        iteration_cnt = 0



        # =========================
        # FedSelect
        # =========================

        if use_fedselect:


            before_params = copy.deepcopy(
                self.trainer.model.state_dict()
            )


            client_mask = (

                global_other_params.get(
                    "client_mask",
                    None
                )

                if global_other_params is not None

                else None

            )


        else:

            client_mask = None



        client_other_params = {}

        train_kwargs = {}



        # =========================
        # FedProx
        # =========================

        if (
            self.args.fedprox
            or
            self.args.scaffold
        ):


            previous_model = copy.deepcopy(
                self.trainer.get_model_params()
            )


            train_kwargs[
                "previous_model"
            ] = previous_model



        # =========================
        # SCAFFOLD
        # =========================

        if self.args.scaffold:


            c_model_global = (
                global_other_params[
                    "c_model_global"
                ]
            )


            for name in c_model_global:

                c_model_global[name] = (
                    c_model_global[name]
                    .to(self.device)
                )



            self.c_model_local.to(
                self.device
            )


            c_model_local = (
                self.c_model_local.state_dict()
            )


            train_kwargs[
                "c_model_global"
            ] = c_model_global


            train_kwargs[
                "c_model_local"
            ] = c_model_local



        # =========================
        # Local optimization
        # =========================

        if use_fedselect and client_mask is not None:

            for p in self.trainer.model.parameters():

                p.requires_grad = True



        for epoch in range(
            self.args.global_epochs_per_round
        ):


            self.construct_mix_dataloader(
                share_data1,
                share_data2,
                share_y,
                round_idx
            )


            self.trainer.train_mix_dataloader(
                epoch,
                self.local_train_mixed_dataloader,
                self.device,
                **train_kwargs
            )


            iteration_cnt += len(
                self.local_train_mixed_dataloader
            )


            logging.debug(
                f"Client {self.client_index}: "
                f"finish local epoch {epoch}"
            )



        # =========================
        # SCAFFOLD update
        # =========================

        if self.args.scaffold:


            iteration_cnt = max(
                iteration_cnt,
                1
            )


            c_new_para = (
                self.c_model_local.state_dict()
            )


            c_delta_para = copy.deepcopy(
                c_new_para
            )


            global_model_para = previous_model


            net_para = (
                self.trainer.model.state_dict()
            )


            if self.trainer.lr_scheduler is not None:

                current_lr = (
                    self.trainer.lr_scheduler.lr
                )

            else:

                current_lr = self.args.lr



            for key in net_para:


                c_new_para[key] = (

                    c_new_para[key]

                    -
                    c_model_global[key]

                    +

                    (
                        global_model_para[key]
                        .to(self.device)
                        -
                        net_para[key]
                    )

                    /
                    (
                        iteration_cnt
                        *
                        current_lr
                    )

                )


                c_delta_para[key] = (

                    c_new_para[key]
                    -
                    c_model_local[key]

                ).cpu()



            self.c_model_local.load_state_dict(
                c_new_para
            )


            self.trainer.model.cpu()

            self.c_model_local.cpu()


            client_other_params[
                "c_delta_para"
            ] = c_delta_para



        # =====================================================
        # FedSelect upload
        # =====================================================

        full_state_after_train = {

            k: v.cpu()

            for k, v in
            self.trainer.model.state_dict().items()

        }



        if use_fedselect and client_mask is not None:


            cond_update = (

                round_idx is not None

                and

                round_idx > 0

                and

                round_idx %
                self.args.lth_epoch_iters
                == 0

            )


            if cond_update:


                from utils.fedselect_utils import (
                    delta_update_mask_layerwise
                )


                client_mask = (
                    delta_update_mask_layerwise(
                        old_params=before_params,
                        new_params=full_state_after_train,
                        mask=client_mask,
                        prune_percent=self.args.prune_percent,
                        prune_target=self.args.prune_target
                    )
                )



            client_other_params[
                "client_mask"
            ] = client_mask



            weights, upload_stats = (
                select_global_params_for_upload(
                    full_state_after_train,
                    client_mask
                )
            )


            model_indexes = list(
                weights.keys()
            )


            client_other_params[
                "upload_stats"
            ] = upload_stats



            logging.debug(
                f"[FedSelect Upload] "
                f"client={self.client_index}, "
                f"ratio={upload_stats['upload_ratio']:.4f}"
            )


        else:


            weights, model_indexes = (
                self.get_model_params()
            )



        client_other_params[
            "client_train_time_sec"
        ] = (
            time.time()
            -
            train_start_time
        )



        return (

            weights,

            model_indexes,

            self.test_data_num,

            client_other_params,

            shared_params_for_simulation

        )



    # ======================================================
    # Trainer interface
    # ======================================================

    def algorithm_on_train(
            self,
            share_data1,
            share_data2,
            share_y,
            round_idx,
            named_params,
            params_type='model',
            global_other_params=None,
            shared_params_for_simulation=None):


        if params_type == 'model':

            self.set_model_params(
                named_params
            )



        return self.fedavg_train(

            share_data1,
            share_data2,
            share_y,

            round_idx,

            global_other_params,

            shared_params_for_simulation

        )