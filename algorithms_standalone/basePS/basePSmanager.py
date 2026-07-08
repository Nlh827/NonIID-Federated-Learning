import logging
import os
import time
import copy
from abc import abstractmethod

import numpy as np
import psutil
import torch

from model.build import create_model

from data_preprocessing.build import load_data

from utils.data_utils import (
    get_selected_clients_label_distribution,
)

from utils.revision_utils import (
    append_csv_row,
    save_json,
    accuracy_auc,
    rounds_to_target,
    cuda_memory_mb,
)



def state_dict_memory_mb(state_dict):
    """
    Calculate model memory size.
    """
    total_bytes = 0

    for _, tensor in state_dict.items():

        if torch.is_tensor(tensor):

            total_bytes += (
                tensor.numel()
                *
                tensor.element_size()
            )

    return total_bytes / 1024.0 / 1024.0



def cpu_memory_mb():

    process = psutil.Process(
        os.getpid()
    )

    return (
        process.memory_info().rss
        /
        1024.0
        /
        1024.0
    )



class BasePSManager(object):

    def __init__(
            self,
            device,
            args):


        self.device = device

        self.args = args



        # =====================================================
        # dataset
        # =====================================================

        self._setup_datasets()



        # =====================================================
        # clients/server
        # =====================================================

        self.selected_clients = None

        self.client_list = []

        self.aggregator = None



        if getattr(
                self.args,
                "instantiate_all",
                True):

            self.number_instantiated_client = (
                self.args.client_num_in_total
            )

        else:

            self.number_instantiated_client = (
                self.args.client_num_per_round
            )



        self._setup_clients()

        self._setup_server()



        self.comm_round = (
            self.args.comm_round
        )



        # =====================================================
        # logs
        # =====================================================

        self.test_acc_list = []

        self.round_log_list = []

        self.client_round_acc_list = []



        if not hasattr(
                self.args,
                "result_dir"):

            self.args.result_dir = os.path.join(

                getattr(
                    self.args,
                    "save_result_dir",
                    "./results_revision"
                ),

                str(self.args.dataset),

                str(self.args.algorithm),

                str(
                    getattr(
                        self.args,
                        "exp_name",
                        "default"
                    )
                ),

                f"seed{self.args.seed}"

            )



        os.makedirs(
            self.args.result_dir,
            exist_ok=True
        )



        # =====================================================
        # prepare VAE proxy data
        # =====================================================

        if getattr(
                self.args,
                "VAE",
                False):

            self._share_data_step()

        else:

            self.global_share_dataset1 = None

            self.global_share_dataset2 = None

            self.global_share_data_y = None





    # =========================================================
    # dataset
    # =========================================================


    def _setup_datasets(self):


        (
            train_data_global_num,
            test_data_global_num,
            train_data_global_dl,
            test_data_global_dl,
            train_data_local_num_dict,
            test_data_local_num_dict,
            test_data_local_dl_dict,
            train_data_local_ori_dict,
            train_targets_local_ori_dict,
            class_num,
            other_params

        ) = load_data(

            load_as="training",

            args=self.args,

            process_id=0,

            mode="standalone",

            task="federated",

            data_efficient_load=True,

            dirichlet_balance=False,

            dirichlet_min_p=None,

            dataset=self.args.dataset,

            datadir=self.args.data_dir,

            partition_method=self.args.partition_method,

            partition_alpha=self.args.partition_alpha,

            client_number=self.args.client_num_in_total,

            batch_size=self.args.batch_size,

            num_workers=self.args.data_load_num_workers,

            data_sampler=self.args.data_sampler,

            resize=self.args.dataset_load_image_size,

            augmentation=self.args.dataset_aug

        )



        self.other_params = other_params


        self.train_data_global_dl = (
            train_data_global_dl
        )

        self.test_data_global_dl = (
            test_data_global_dl
        )


        self.train_data_global_num = (
            train_data_global_num
        )

        self.test_data_global_num = (
            test_data_global_num
        )



        self.test_data_local_dl_dict = (
            test_data_local_dl_dict
        )


        self.train_data_local_num_dict = (
            train_data_local_num_dict
        )


        self.test_data_local_num_dict = (
            test_data_local_num_dict
        )


        self.train_data_local_ori_dict = (
            train_data_local_ori_dict
        )


        self.train_targets_local_ori_dict = (
            train_targets_local_ori_dict
        )



        self.client_dataidx_map = (
            other_params["client_dataidx_map"]
        )


        self.train_cls_local_counts_dict = (
            other_params["train_cls_local_counts_dict"]
        )


        self.class_num = class_num



        # fill missing classes

        if (
            self.train_cls_local_counts_dict
            is not None
        ):


            classes = list(
                range(self.class_num)
            )


            for key in self.train_cls_local_counts_dict:


                current = (
                    self.train_cls_local_counts_dict[key]
                )


                if len(current) != len(classes):

                    missing = (
                        set(classes)
                        -
                        set(current)
                    )


                    for c in missing:

                        current[c] = 0



        else:

            self.train_cls_local_counts_dict = None




    # =========================================================
    # abstract setup
    # =========================================================


    def _setup_server(self):

        pass



    def _setup_clients(self):

        pass


    # =========================================================
    # VAE proxy preparation
    # =========================================================


    def _share_data_step(self):

        """
        VAE federated pre-training stage.

        Flow:

        client VAE training
              |
              v
        VAE aggregation
              |
              v
        distribute global VAE
              |
              v
        generate proxy data
              |
              v
        collect shared proxy data
        """


        if not getattr(
                self.args,
                "VAE",
                False):

            return



        for round_idx in range(
                self.args.VAE_comm_round
        ):


            logging.info(
                f"############ VAE Round {round_idx} ############"
            )


            client_indexes = (
                self.client_sample_for_VAE(
                    round_idx,
                    self.args.client_num_in_total,
                    self.args.VAE_client_num_per_round
                )
            )



            vae_states = []



            for cid in client_indexes:


                client = (
                    self.client_list[cid]
                )


                local_state = (
                    client.train_vae_model()
                )



                if local_state is not None:


                    vae_states.append(

                        (

                            client.local_sample_number,

                            local_state

                        )

                    )



            # ===============================
            # aggregate VAE
            # ===============================

            if len(vae_states) > 0:


                self.aggregator.aggregate_vae(
                    vae_states
                )



                global_vae_state = (
                    self.aggregator.get_vae_param()
                )



                # distribute global VAE

                if global_vae_state is not None:


                    for client in self.client_list:

                        client.set_vae_para(
                            global_vae_state
                        )



            # optional VAE evaluation

            if hasattr(
                    self.aggregator,
                    "test_on_server_by_vae"):

                self.aggregator.test_on_server_by_vae(
                    round_idx
                )



        # =====================================================
        # Generate proxy data after VAE convergence
        # =====================================================


        self._get_local_shared_data()



        if hasattr(
                self.aggregator,
                "save_vae_param"):

            self.aggregator.save_vae_param()





    # =========================================================
    # VAE client sampling
    # =========================================================


    def client_sample_for_VAE(
            self,
            round_idx,
            client_num_in_total,
            client_num_per_round):


        if (
            client_num_in_total
            ==
            client_num_per_round
        ):


            client_indexes = list(
                range(client_num_in_total)
            )


        else:


            rng = np.random.default_rng(

                int(self.args.seed)
                +
                int(round_idx)

            )


            client_indexes = rng.choice(

                np.arange(
                    client_num_in_total
                ),

                size=min(

                    client_num_per_round,

                    client_num_in_total

                ),

                replace=False

            ).tolist()



        logging.info(
            "VAE selected clients: %s",
            str(client_indexes)
        )


        return client_indexes





    # =========================================================
    # collect proxy data
    # =========================================================


    def _get_local_shared_data(self):


        if not getattr(
                self.args,
                "VAE",
                False):

            return



        shared_data1 = []

        shared_data2 = []

        shared_targets = []



        for client in self.client_list:


            result = (
                client.generate_proxy_data()
            )


            if result is None:

                continue



            data1, data2, labels = result



            if data1 is None:

                continue



            shared_data1.append(
                data1
            )


            shared_data2.append(
                data2
            )


            shared_targets.append(
                labels
            )



        if len(shared_data1) == 0:


            self.global_share_dataset1 = None

            self.global_share_dataset2 = None

            self.global_share_data_y = None


            self.proxy_data_mb = 0.0

            return





        self.global_share_dataset1 = torch.cat(
            shared_data1,
            dim=0
        )


        self.global_share_dataset2 = torch.cat(
            shared_data2,
            dim=0
        )


        self.global_share_data_y = torch.cat(
            shared_targets,
            dim=0
        )



        # ===============================
        # proxy communication accounting
        # ===============================


        proxy_bytes = (

            self.global_share_dataset1.numel()
            *
            self.global_share_dataset1.element_size()

            +

            self.global_share_dataset2.numel()
            *
            self.global_share_dataset2.element_size()

            +

            self.global_share_data_y.numel()
            *
            self.global_share_data_y.element_size()

        )



        self.proxy_data_mb = (
            proxy_bytes
            /
            1024.0
            /
            1024.0
        )



        logging.info(

            f"[Proxy] shared proxy size "
            f"{self.proxy_data_mb:.4f} MB"

        )





    # =========================================================
    # state initialization
    # =========================================================


    def get_init_state_kargs(self):


        self.selected_clients = list(

            range(
                self.args.client_num_in_total
            )

        )


        return {}





    def get_update_state_kargs(self):


        if self.args.loss_fn in [
            "LDAMLoss",
            "FocalLoss",
            "local_FocalLoss",
            "local_LDAMLoss"
        ]:


            self.selected_clients_label_distribution = (
                get_selected_clients_label_distribution(
                    self.local_cls_num_list_dict,
                    self.class_num,
                    self.selected_clients,
                    min_limit=1
                )
            )


            return {

                "weight": None,

                "selected_cls_num_list":
                    self.selected_clients_label_distribution,

                "local_cls_num_list_dict":
                    self.local_cls_num_list_dict

            }



        return {}


    # =========================================================
    # learning rate schedule
    # =========================================================


    def lr_schedule(
            self,
            num_iterations,
            warmup_epochs):


        epochs = (
            self.server_timer.global_outer_epoch_idx
        )

        iterations = (
            self.server_timer.global_outer_iter_idx
        )


        if self.args.sched == "no":

            return


        if epochs < warmup_epochs:

            self.aggregator.trainer.warmup_lr_schedule(
                iterations
            )

        else:

            if (
                iterations > 0
                and
                iterations % num_iterations == 0
            ):

                self.aggregator.trainer.lr_schedule(
                    epochs
                )




    # =========================================================
    # main federated training loop
    # =========================================================


    def train(self):


        round_log_path = os.path.join(
            self.args.result_dir,
            "round_logs.csv"
        )


        summary_path = os.path.join(
            self.args.result_dir,
            "summary.json"
        )



        for round_idx in range(
                self.comm_round
        ):


            round_start_time = time.time()


            if torch.cuda.is_available():

                torch.cuda.reset_peak_memory_stats()



            logging.info(
                f"############ Communication Round "
                f"{round_idx} ############"
            )



            # =====================================================
            # personalization-only stage
            # =====================================================


            if getattr(
                    self.args,
                    "local_personalize",
                    False):


                start_round = int(

                    self.args.comm_round
                    *
                    float(
                        getattr(
                            self.args,
                            "local_personalize_start_frac",
                            0.8
                        )
                    )

                )


                if round_idx >= start_round:


                    logging.info(

                        f"Personalization stage "
                        f"round={round_idx}"

                    )



                    local_accs = []


                    start_time = time.time()



                    for cid in range(
                            len(self.client_list)
                    ):


                        client = (
                            self.client_list[cid]
                        )


                        client.personalize_after_round(
                            round_idx
                        )


                        acc = client.test(
                            self.args.VAE_comm_round
                            +
                            round_idx
                        )


                        local_accs.append(
                            float(acc)
                        )



                    avg_acc = (

                        sum(local_accs)
                        /
                        len(local_accs)

                        if len(local_accs)>0
                        else 0.0

                    )


                    self.test_acc_list.append(
                        avg_acc
                    )



                    row = {


                        "dataset":
                            self.args.dataset,


                        "algorithm":
                            self.args.algorithm,


                        "seed":
                            int(self.args.seed),


                        "round":
                            int(round_idx),


                        "server_acc":
                            avg_acc,


                        "eval_scope":
                            "personalized_local_avg",


                        "auc_so_far":
                            accuracy_auc(
                                self.test_acc_list
                            ),


                        "round_time_sec":
                            time.time()
                            -
                            round_start_time,


                        "stage":
                            "personalization_only",

                        "round_total_comm_mb":
                            0.0,

                        "model_upload_mb":
                            0.0,

                        "model_download_mb":
                            0.0,

                        "proxy_memory_mb":
                            float(
                                getattr(
                                    self,
                                    "proxy_data_mb",
                                    0.0
                                )
                            )

                    }



                    self.round_log_list.append(
                        copy.deepcopy(row)
                    )


                    if getattr(
                            self.args,
                            "save_round_logs",
                            True):

                        append_csv_row(
                            round_log_path,
                            row
                        )



                    continue





            # =====================================================
            # normal federated stage
            # =====================================================


            if round_idx == 0:


                named_params = (
                    self.aggregator
                    .get_global_model_params()
                )


                params_type = "model"


                global_other_params = {}


                shared_params_for_simulation = {}



                if getattr(
                        self.args,
                        "scaffold",
                        False):


                    global_other_params[
                        "c_model_global"
                    ] = (
                        self.aggregator
                        .c_model_global
                        .state_dict()
                    )



            client_indexes = (
                self.aggregator.client_sampling(
                    round_idx,
                    self.args.client_num_in_total,
                    self.args.client_num_per_round
                )
            )



            update_state_kargs = (
                self.get_update_state_kargs()
            )



            (
                named_params,
                params_type,
                global_other_params,
                shared_params_for_simulation

            ) = self.algorithm_train(

                round_idx,

                client_indexes,

                named_params,

                params_type,

                global_other_params,

                update_state_kargs,

                shared_params_for_simulation

            )



            avg_acc = (
                self.aggregator
                .test_on_server_for_round(
                    self.args.VAE_comm_round
                    +
                    round_idx
                )
            )



            self.test_acc_list.append(
                float(avg_acc)
            )



            auc_value = accuracy_auc(
                self.test_acc_list
            )



            comm_stats = getattr(
                self.aggregator,
                "round_comm_stats",
                {}
            )



            row = {


                "dataset":
                    self.args.dataset,


                "algorithm":
                    self.args.algorithm,


                "seed":
                    int(self.args.seed),


                "round":
                    int(round_idx),


                "server_acc":
                    float(avg_acc),


                "eval_scope":
                    "global_test",


                "auc_so_far":
                    float(auc_value),


                "round_time_sec":
                    time.time()
                    -
                    round_start_time,


                "selected_clients":
                    comm_stats.get(
                        "selected_clients",
                        str(client_indexes)
                    ),


                "num_selected_clients":
                    len(client_indexes),



                "model_upload_mb":
                    comm_stats.get(
                        "model_upload_mb",
                        0.0
                    ),


                "model_download_mb":
                    comm_stats.get(
                        "model_download_mb",
                        0.0
                    ),


                "round_total_comm_mb":
                    comm_stats.get(
                        "total_comm_mb",
                        0.0
                    ),



                "proxy_memory_mb":
                    float(
                        getattr(
                            self,
                            "proxy_data_mb",
                            0.0
                        )
                    ),



                "stage":
                    "federated"

            }



            self.round_log_list.append(
                copy.deepcopy(row)
            )


            if getattr(
                    self.args,
                    "save_round_logs",
                    True):

                append_csv_row(
                    round_log_path,
                    row
                )



            print(
                f"[Round {round_idx}] "
                f"acc={avg_acc:.4f}, "
                f"AUC={auc_value:.4f}"
            )




        # =====================================================
        # final summary
        # =====================================================


        summary = {


            "dataset":
                self.args.dataset,


            "algorithm":
                self.args.algorithm,


            "seed":
                int(self.args.seed),


            "final_acc":
                float(
                    self.test_acc_list[-1]
                )
                if len(self.test_acc_list)>0
                else 0.0,


            "best_acc":
                float(
                    max(self.test_acc_list)
                )
                if len(self.test_acc_list)>0
                else 0.0,


            "auc":
                float(
                    accuracy_auc(
                        self.test_acc_list
                    )
                ),


            "comm_round":
                int(self.comm_round),


            "proxy_data_mb_once":
                float(
                    getattr(
                        self,
                        "proxy_data_mb",
                        0.0
                    )
                ),


            "total_comm_mb":
                float(
                    sum(
                        x.get(
                            "round_total_comm_mb",
                            0.0
                        )

                        for x in self.round_log_list
                    )
                )

        }



        save_json(
            summary_path,
            summary
        )



        self.aggregator.save_classifier()





    # =========================================================
    # algorithm interface
    # =========================================================


    @abstractmethod
    def algorithm_train(
            self,
            round_idx,
            client_indexes,
            named_params,
            params_type,
            global_other_params,
            update_state_kargs,
            shared_params_for_simulation):

        pass