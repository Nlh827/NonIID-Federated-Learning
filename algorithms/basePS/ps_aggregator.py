import copy
import logging
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

from model.FL_VAE import *

from utils.tool import (
    AverageMeter,
    accuracy,
    generate_reconst_images
)

from utils.log_info import log_info

from utils.data_utils import (
    average_named_params,
    check_type
)

from utils.revision_utils import (
    state_dict_bytes,
    mask_bytes,
    bytes_to_mb
)



class PSAggregator(object):

    def __init__(
            self,
            train_dataloader,
            test_dataloader,
            train_data_num,
            test_data_num,
            train_data_local_num_dict,
            worker_num,
            device,
            args,
            model_trainer,
            vae_model):


        self.trainer = model_trainer


        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader


        self.train_data_num = train_data_num
        self.test_data_num = test_data_num


        self.train_data_local_num_dict = (
            train_data_local_num_dict
        )


        self.pre_model_parms = (
            self.get_global_model_params()
        )


        self.worker_num = worker_num

        self.device = device

        self.args = args



        # client updates

        self.model_dict = {}

        self.grad_dict = {}

        self.sample_num_dict = {}



        # FedSelect cache

        self.latest_client_params = {}

        self.latest_client_masks = {}



        self.client_other_params_dict = {}



        # VAE

        self.vae_model = vae_model



        # upload flag

        self.flag_client_model_uploaded_dict = {}

        for idx in range(self.worker_num):

            self.flag_client_model_uploaded_dict[idx] = False



        self.selected_clients = None



        # revision statistics

        self.round_comm_stats = {}

        self.round_cost_history = []

        self.last_aggregate_time_sec = 0.0



    # ======================================================
    # model interface
    # ======================================================


    def get_global_model_params(self):

        return self.trainer.get_model_params()



    def get_global_generator(self):

        if hasattr(
                self.trainer,
                "get_generator"
        ):

            return self.trainer.get_generator()

        return None



    def set_global_model_params(
            self,
            model_parameters):

        self.trainer.set_model_params(
            model_parameters
        )



    def set_grad_params(
            self,
            named_grads):

        self.trainer.set_grad_params(
            named_grads
        )



    def clear_grad_params(self):

        self.trainer.clear_grad_params()



    def update_model_with_grad(self):

        self.trainer.update_model_with_grad()



    # ======================================================
    # VAE interface
    # ======================================================


    def get_vae_param(self):

        if self.vae_model is None:

            return None


        return deepcopy(
            self.vae_model.cpu()
            .state_dict()
        )



    def set_vae_param(
            self,
            para_dict):


        if self.vae_model is None:

            return


        self.vae_model.load_state_dict(
            para_dict
        )



    def aggregate_vae(
            self,
            vae_state_list):

        """
        Federated averaging for VAE.

        vae_state_list:
            [
              (sample_num, state_dict),
              ...
            ]
        """


        if (
            self.vae_model is None
            or
            len(vae_state_list)==0
        ):

            return None



        total_num = sum(
            x[0]
            for x in vae_state_list
        )


        avg_state = {}


        for key in vae_state_list[0][1]:

            avg_state[key] = sum(

                x[0]
                *
                x[1][key]

                for x in vae_state_list

            ) / total_num



        self.vae_model.load_state_dict(
            avg_state
        )


        return avg_state



    def save_classifier(self):

        torch.save(
            self.trainer.model,
            "classifier_model.pth"
        )



    def save_vae_param(self):

        if self.vae_model is not None:

            torch.save(
                self.vae_model,
                "vae_model.pth"
            )



    def get_generate_model_classifier_para(self):

        if self.vae_model is None:

            return None


        return deepcopy(
            self.vae_model
            .get_classifier()
            .cpu()
            .state_dict()
        )
    # ======================================================
    # receive client update
    # ======================================================


    def add_local_trained_result(
            self,
            index,
            model_params,
            model_indexes,
            sample_num,
            client_other_params=None):


        logging.info(
            "add_model index=%d",
            index
        )


        self.model_dict[index] = model_params

        self.sample_num_dict[index] = sample_num


        self.client_other_params_dict[index] = (
            client_other_params
            if client_other_params is not None
            else {}
        )


        self.flag_client_model_uploaded_dict[index] = True



        # FedSelect cache

        self.latest_client_params[index] = (
            copy.deepcopy(model_params)
        )


        if (
            client_other_params is not None
            and
            "client_mask" in client_other_params
        ):

            self.latest_client_masks[index] = (
                client_other_params["client_mask"]
            )



        # communication statistics

        upload_bytes = (
            state_dict_bytes(model_params)
        )


        mask_size = mask_bytes(
            self.latest_client_masks.get(index,None)
        )


        if not hasattr(
                self,
                "current_round_client_costs"
        ):

            self.current_round_client_costs = {}



        train_time = 0.0


        if client_other_params:

            train_time = float(
                client_other_params.get(
                    "client_train_time_sec",
                    0.0
                )
            )



        self.current_round_client_costs[index] = {

            "client_id":
                int(index),

            "sample_num":
                int(sample_num),

            "model_upload_mb":
                bytes_to_mb(upload_bytes),

            "mask_upload_mb":
                bytes_to_mb(mask_size),

            "client_train_time_sec":
                train_time

        }



    def get_client_latest_params(
            self,
            client_index):

        return self.latest_client_params[client_index]



    def get_global_model(self):

        return self.trainer.get_model()



    # ======================================================
    # client sampling
    # ======================================================


    def client_sampling(
            self,
            round_idx,
            client_num_in_total,
            client_num_per_round):


        if client_num_in_total == client_num_per_round:


            clients = list(
                range(client_num_in_total)
            )


        else:


            seed = (
                int(getattr(self.args,"seed",0))
                +
                int(round_idx)
            )


            rng = np.random.default_rng(
                seed
            )


            clients = rng.choice(

                np.arange(client_num_in_total),

                min(
                    client_num_per_round,
                    client_num_in_total
                ),

                replace=False

            ).tolist()



        clients = [
            int(x)
            for x in clients
        ]


        logging.info(
            "sampling clients=%s",
            str(clients)
        )


        self.selected_clients = clients


        return clients




    # ======================================================
    # test
    # ======================================================


    def test_on_server_for_all_clients(
            self,
            epoch,
            tracker=None,
            metrics=None):


        return self.trainer.test(

            epoch,

            self.test_dataloader,

            self.device

        )



    def test_on_server_for_round(
            self,
            round):


        return self.trainer.test_on_server_for_round(

            round,

            self.test_dataloader,

            self.device

        )



    # ======================================================
    # aggregation
    # ======================================================


    def aggregate(self):


        start = time.time()


        global_other_params = {}

        shared_params_for_simulation = {}



        model_list = []

        sample_list = []



        for cid in self.selected_clients:


            model_list.append(

                (

                    self.sample_num_dict[cid],

                    self.model_dict[cid]

                )

            )


            sample_list.append(
                self.sample_num_dict[cid]
            )



        average_weights_dict_list,_ = (
            self.trainer.averager
            .get_average_weight(
                sample_list
            )
        )



        # ==================================================
        # FedSelect aggregation
        # ==================================================


        if getattr(
                self.args,
                "fedselect",
                False):


            global_params = copy.deepcopy(
                self.pre_model_parms
            )


            for name in global_params:


                value = None

                total_weight = 0



                for cid in self.selected_clients:


                    client_params = (
                        self.latest_client_params
                        .get(cid,{})
                    )


                    if name not in client_params:

                        continue



                    weight = (
                        self.sample_num_dict[cid]
                    )


                    tensor = (
                        client_params[name]
                        .to(global_params[name].device)
                    )


                    if value is None:

                        value = (
                            weight*tensor
                        )

                    else:

                        value += (
                            weight*tensor
                        )


                    total_weight += weight



                if total_weight>0:

                    global_params[name] = (
                        value / total_weight
                    )



            averaged_params = global_params



        else:


            averaged_params = average_named_params(

                model_list,

                average_weights_dict_list

            )



        # ==================================================
        # SCAFFOLD
        # ==================================================


        if getattr(
                self.args,
                "scaffold",
                False):


            deltas = []


            for cid in self.selected_clients:


                other = (
                    self.client_other_params_dict[cid]
                )


                deltas.append(
                    other["c_delta_para"]
                )



            avg_delta = deepcopy(
                deltas[0]
            )


            for k in avg_delta:

                avg_delta[k].zero_()



            for delta in deltas:

                for k in avg_delta:

                    avg_delta[k] += (
                        delta[k]
                        /
                        len(deltas)
                    )



            c_global = (
                self.c_model_global
                .state_dict()
            )


            for k in c_global:

                c_global[k] += check_type(

                    avg_delta[k],

                    c_global[k].type()

                )



            self.c_model_global.load_state_dict(
                c_global
            )


            global_other_params[
                "c_model_global"
            ] = c_global



        # update server model

        self.set_global_model_params(
            averaged_params
        )


        self.pre_model_parms = copy.deepcopy(
            averaged_params
        )



        # ==================================================
        # communication statistics
        # ==================================================

        upload = 0.0


        for cid in self.selected_clients:


            stats = (
                self.client_other_params_dict
                .get(cid,{})
                .get("upload_stats",{})
            )


            upload += float(
                stats.get(
                    "uploaded_model_mb",
                    0
                )
            )



        self.round_comm_stats = {

            "selected_clients":
                str(self.selected_clients),

            "num_selected_clients":
                len(self.selected_clients),

            "model_upload_mb":
                upload,

            "model_download_mb":
                upload,

            "total_comm_mb":
                upload*2,

        }



        self.last_aggregate_time_sec = (
            time.time()-start
        )


        self.round_comm_stats[
            "aggregate_time_sec"
        ] = self.last_aggregate_time_sec



        self.round_cost_history.append(
            copy.deepcopy(
                self.round_comm_stats
            )
        )



        return (

            averaged_params,

            global_other_params,

            shared_params_for_simulation

        )




    # ======================================================
    # Generate proxy data
    # ======================================================


    def server_generate_data_by_vae(self):


        if self.vae_model is None:

            return



        transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )


        dataset_name = (
            self.args.dataset.lower()
        )


        if dataset_name == "svhn":


            dataset = torchvision.datasets.SVHN(

                self.args.data_dir,

                split="train",

                download=True,

                transform=transform

            )


        elif dataset_name == "cifar10":


            dataset = torchvision.datasets.CIFAR10(

                self.args.data_dir,

                train=True,

                download=True,

                transform=transform

            )


        elif dataset_name in [
            "fmnist",
            "fashionmnist"
        ]:


            dataset = torchvision.datasets.FashionMNIST(

                self.args.data_dir,

                train=True,

                download=True,

                transform=transform

            )


        else:

            raise NotImplementedError(
                f"Unsupported dataset {self.args.dataset}"
            )



        loader = torch.utils.data.DataLoader(

            dataset,

            batch_size=self.args.VAE_batch_size,

            shuffle=False

        )



        self.vae_model.to(
            self.device
        )


        self.vae_model.eval()



        with torch.no_grad():


            for idx,(x,y) in enumerate(loader):


                x = x.to(
                    self.device
                )

                y = y.to(
                    self.device
                )



                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    rx1,
                    rx2

                ) = self.vae_model(x)



                if idx==0:


                    self.global_share_dataset1 = rx1

                    self.global_share_dataset2 = rx2

                    self.global_share_data_y = y



                else:


                    self.global_share_dataset1 = torch.cat(
                        [
                            self.global_share_dataset1,
                            rx1
                        ]
                    )


                    self.global_share_dataset2 = torch.cat(
                        [
                            self.global_share_dataset2,
                            rx2
                        ]
                    )


                    self.global_share_data_y = torch.cat(
                        [
                            self.global_share_data_y,
                            y
                        ]
                    )



        return (
            self.global_share_dataset1,
            self.global_share_dataset2,
            self.global_share_data_y
        )