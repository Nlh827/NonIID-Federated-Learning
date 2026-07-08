import logging
import math
import os
import sys
import random
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset, Subset

from algorithms.basePS.ps_client_trainer import PSTrainer

from utils.data_utils import optimizer_to
from utils.tool import *
from utils.set import *
from utils.log_info import log_info

from model.FL_VAE import *

from optim.AdamW import AdamW

from data_preprocessing.cifar10.datasets import (
    Dataset_Personalize,
    Dataset_3Types_ImageData
)

import torchvision.transforms as transforms

from utils.randaugment4fixmatch import (
    RandAugmentMC,
    RandAugment_no_CutOut,
    Cutout
)

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.getcwd(), "../../../"))
)


class XYDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y



class Client(PSTrainer):

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
            dataset_num
    ):

        super().__init__(
            client_index,
            train_ori_data,
            train_ori_targets,
            test_dataloader,
            train_data_num,
            test_data_num,
            device,
            args,
            model_trainer
        )


        self.args = args
        self.device = device

        self.test_dataloader = test_dataloader

        self.train_ori_data = train_ori_data
        self.train_ori_targets = train_ori_targets

        self.train_cls_counts_dict = train_cls_counts_dict

        self.dataset_num = dataset_num


        # ===============================
        # VAE initialization
        # ===============================

        self.vae_model = None
        self.vae_optimizer = None

        if self.args.VAE and vae_model is not None:

            logging.info(
                f"client {self.client_index}: VAE model initialized"
            )

            self.vae_model = vae_model

            self.vae_optimizer = AdamW(
                [
                    {
                        "params": self.vae_model.parameters()
                    }
                ],
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-6
            )


        # ===============================
        # local training information
        # ===============================

        self.local_num_iterations = math.ceil(
            len(self.train_ori_data)
            /
            self.args.batch_size
        )


        # proxy data cache
        self.local_share_data1 = None
        self.local_share_data2 = None
        self.local_share_data_y = None

        self.global_share_data1 = None
        self.global_share_data2 = None
        self.global_share_y = None


        # statistics
        self.client_stats = {}

        self.last_entropy_selected_count = 0
        self.last_entropy_strategy = None
        self.last_entropy_select_ratio = 0


        self.personal_head_state = None
        self.personalized_model_state = None



        # ===============================
        # SCAFFOLD
        # ===============================

        if self.args.scaffold:

            from model.build import create_model

            self.c_model_local = create_model(
                self.args,
                model_name=self.args.model,
                output_dim=self.args.model_output_dim
            )

            for _, params in self.c_model_local.named_parameters():
                params.data = params.data * 0



        # local dataloader
        self._construct_train_ori_dataloader()



        # Non-IID property estimation
        if self.args.VAE and getattr(
                self.args,
                "VAE_adaptive",
                False
        ):

            self._set_local_traindata_property()

            logging.info(
                f"client {self.client_index} property: "
                f"{self.local_traindata_property}"
            )



    # ======================================================
    # model helper
    # ======================================================

    def _get_client_model(self):


        if hasattr(self, "trainer"):
            return self.trainer.model


        if hasattr(self, "model"):
            return self.model


        if hasattr(self, "model_trainer"):
            return self.model_trainer.model


        if hasattr(self, "net"):
            return self.net


        raise AttributeError(
            "Cannot find client model"
        )


    # ======================================================
    # personalization helper
    # ======================================================

    def _is_head_key(self, k):

        return (
            ".fc." in k
            or k.startswith("fc.")
            or ".classifier." in k
            or k.startswith("classifier.")
            or ".head." in k
            or k.startswith("head.")
            or ".linear." in k
            or k.startswith("linear.")
        )


    def _extract_head_state(self, model):

        state = model.state_dict()

        return {
            k:v.detach().cpu().clone()
            for k,v in state.items()
            if self._is_head_key(k)
        }


    def _load_head_state(
            self,
            model,
            head_state
    ):

        if head_state:

            model.load_state_dict(
                head_state,
                strict=False
            )



    # ======================================================
    # uncertainty selection
    # ======================================================

    @torch.no_grad()
    def _select_hard_indices_by_margin(
            self,
            model,
            dataset,
            device,
            select_ratio=0.3,
            batch_size=256
    ):

        model.eval()

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )


        margins = []

        for x,_ in loader:

            x = x.to(device)

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )

            top2 = torch.topk(
                probs,
                k=2,
                dim=1
            ).values

            margin = (
                top2[:,0]
                -
                top2[:,1]
            )

            margins.append(
                margin.cpu()
            )


        margins = torch.cat(
            margins
        )


        n = len(dataset)

        if n == 0:
            return []


        order = torch.argsort(
            margins,
            descending=False
        )


        k = max(
            1,
            int(select_ratio*n)
        )


        return order[:k].tolist()

    # ======================================================
    # Personalization
    # ======================================================

    def personalize_after_round(self, round_idx):

        """
        最后阶段本地个性化微调

        由 FedAVGManager 控制调用时机
        """

        if not getattr(
                self.args,
                "local_personalize",
                False):

            return


        model = self._get_client_model()


        if self.personalized_model_state is None:

            self.personalized_model_state = {
                k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }



        model.train()


        steps = int(
            getattr(
                self.args,
                "local_personalize_steps",
                10
            )
        )


        lr = float(
            getattr(
                self.args,
                "local_personalize_lr",
                5e-5
            )
        )


        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9
        )



        for _ in range(steps):

            for x,y in self.train_ori_dataloader:

                x = x.to(self.device)

                y = y.to(self.device)


                optimizer.zero_grad()


                out = model(x)


                loss = F.cross_entropy(
                    out,
                    y.long()
                )


                loss.backward()


                optimizer.step()



        logging.info(
            f"client {self.client_index}: "
            f"personalization finished at round {round_idx}"
        )




    # ======================================================
    # Original local dataset
    # ======================================================

    def _construct_train_ori_dataloader(self):

        dataset = XYDataset(
            self.train_ori_data,
            self.train_ori_targets
        )


        self.train_ori_dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )



    # ======================================================
    # Local data property
    # ======================================================

    def _set_local_traindata_property(self):

        counts = self.train_cls_counts_dict


        if counts is None:

            self.local_traindata_property = None

            return


        values = list(
            counts.values()
        )


        if len(values) == 0:

            self.local_traindata_property = 0

            return



        values = np.array(
            values
        )


        self.local_traindata_property = (
            np.std(values)
            /
            (np.mean(values)+1e-8)
        )



    # ======================================================
    # VAE training
    # ======================================================

    def train_vae_model(self):

        """
        Client-side VAE training

        输出:
            VAE local update
        """


        if (
            not getattr(
                self.args,
                "VAE",
                False
            )
            or
            self.vae_model is None
        ):

            return None



        self.vae_model.to(
            self.device
        )


        self.vae_model.train()



        loader = DataLoader(

            XYDataset(
                self.train_ori_data,
                self.train_ori_targets
            ),

            batch_size=self.args.VAE_batch_size,

            shuffle=True

        )



        criterion = torch.nn.MSELoss()



        epochs = int(
            getattr(
                self.args,
                "VAE_local_epoch",
                1
            )
        )



        for epoch in range(epochs):


            total_loss = 0


            for x,_ in loader:


                x = x.to(
                    self.device
                )


                self.vae_optimizer.zero_grad()



                output = self.vae_model(
                    x
                )


                if output is None:

                    continue



                _, _, recon, _, _, _, _, _ = output



                loss = criterion(
                    recon,
                    x
                )


                loss.backward()


                self.vae_optimizer.step()



                total_loss += loss.item()



            logging.debug(

                f"client {self.client_index} "
                f"VAE epoch {epoch}: "
                f"{total_loss:.4f}"

            )



        return self.vae_model.state_dict()




    # ======================================================
    # Generate proxy data from VAE
    # ======================================================

    @torch.no_grad()
    def generate_proxy_data(
            self,
            num_samples=None
    ):


        if (
            self.vae_model is None
            or
            not getattr(
                self.args,
                "VAE",
                False
            )
        ):

            return None,None,None



        self.vae_model.eval()


        if num_samples is None:

            num_samples = min(
                len(self.train_ori_data),
                self.args.batch_size
            )



        data = self.train_ori_data[
            :num_samples
        ]



        data = data.to(
            self.device
        )



        output = self.vae_model(
            data
        )



        if output is None:

            return None,None,None



        (
            _,
            _,
            xi,
            _,
            _,
            rx,
            rx_noise1,
            rx_noise2
        ) = output



        mode = getattr(
            self.args,
            "proxy_mode",
            "noisy_residual"
        )



        if mode == "noisy_residual":

            proxy1 = rx_noise1

            proxy2 = rx_noise2



        elif mode == "residual":

            proxy1 = rx

            proxy2 = rx



        elif mode == "reconstruction":

            proxy1 = xi

            proxy2 = xi



        elif mode == "raw":

            proxy1 = data

            proxy2 = data



        else:

            raise ValueError(
                f"Unknown proxy mode {mode}"
            )



        labels = self.train_ori_targets[
            :num_samples
        ].to(
            self.device
        )



        return (
            proxy1.cpu(),
            proxy2.cpu(),
            labels.cpu()
        )

    # ======================================================
    # Entropy based proxy selection
    # ======================================================

    @torch.no_grad()
    def filter_dataset_by_entropy(
            self,
            data,
            targets,
            select_ratio=None
    ):

        """
        Entropy based sample selection.

        high_entropy:
            select uncertain samples

        low_entropy:
            select confident samples

        random:
            random selection
        """

        strategy = getattr(
            self.args,
            "entropy_selection_strategy",
            "none"
        )


        if (
            strategy == "none"
            or
            data is None
            or
            len(data) == 0
        ):

            return data, targets



        if select_ratio is None:

            select_ratio = float(
                getattr(
                    self.args,
                    "entropy_select_ratio",
                    0.6
                )
            )



        num_select = max(
            1,
            int(len(data) * select_ratio)
        )


        # -----------------------------
        # random selection
        # -----------------------------

        if strategy == "random":

            indices = torch.randperm(
                len(data)
            )[:num_select]



        else:

            model = self._get_client_model()

            model.eval()


            dataset = XYDataset(
                data,
                targets
            )


            loader = DataLoader(
                dataset,
                batch_size=self.args.entropy_batch_size,
                shuffle=False
            )


            entropy_list = []


            for x, _ in loader:


                x = x.to(
                    self.device
                )


                logits = model(x)


                prob = torch.softmax(
                    logits,
                    dim=1
                )


                entropy = -torch.sum(

                    prob
                    *
                    torch.log(
                        prob + 1e-12
                    ),

                    dim=1

                )


                entropy_list.append(
                    entropy.cpu()
                )



            entropy = torch.cat(
                entropy_list
            )



            if strategy in [
                "high_entropy",
                "class_balanced_entropy"
            ]:


                indices = torch.argsort(
                    entropy,
                    descending=True
                )[:num_select]


            elif strategy == "low_entropy":


                indices = torch.argsort(
                    entropy,
                    descending=False
                )[:num_select]


            else:

                indices = torch.randperm(
                    len(data)
                )[:num_select]



        self.last_entropy_selected_count = len(indices)

        self.last_entropy_strategy = strategy

        self.last_entropy_select_ratio = (
            len(indices)
            /
            len(data)
        )



        self.client_stats.update(

            {

                "entropy_strategy":
                    strategy,

                "entropy_selected":
                    len(indices),

                "entropy_ratio":
                    self.last_entropy_select_ratio

            }

        )



        return (
            data[indices],
            targets[indices]
        )





    # ======================================================
    # Construct mixed dataloader
    # ======================================================

    def construct_mix_dataloader(
            self,
            share_data1,
            share_data2,
            share_y,
            round_idx=None
    ):


        local_data = self.train_ori_data

        local_targets = self.train_ori_targets



        # --------------------------------------
        # No proxy data
        # --------------------------------------

        if (
            not getattr(
                self.args,
                "use_proxy_data",
                True
            )
            or
            share_data1 is None
        ):


            dataset = XYDataset(
                local_data,
                local_targets
            )


            self.local_train_mixed_dataloader = DataLoader(

                dataset,

                batch_size=self.args.batch_size,

                shuffle=True

            )


            return self.local_train_mixed_dataloader




        # --------------------------------------
        # entropy selection
        # --------------------------------------

        proxy1 = share_data1

        proxy2 = share_data2

        proxy_y = share_y



        proxy1, proxy_y1 = (
            self.filter_dataset_by_entropy(
                proxy1,
                proxy_y
            )
        )


        proxy2, proxy_y2 = (
            self.filter_dataset_by_entropy(
                proxy2,
                proxy_y
            )
        )



        data1 = torch.cat(

            [
                local_data,
                proxy1,
                proxy2
            ],

            dim=0

        )


        target1 = torch.cat(

            [
                local_targets,
                proxy_y1,
                proxy_y2
            ],

            dim=0

        )



        dataset = XYDataset(
            data1,
            target1
        )



        self.local_train_mixed_dataloader = DataLoader(

            dataset,

            batch_size=self.args.batch_size,

            shuffle=True

        )



        return self.local_train_mixed_dataloader





    # ======================================================
    # Client train entry
    # ======================================================

    def train(
            self,
            share_data1,
            share_data2,
            share_y,
            round_idx,
            named_params,
            params_type="model",
            global_other_params=None,
            shared_params_for_simulation=None
    ):


        return self.algorithm_on_train(

            share_data1,

            share_data2,

            share_y,

            round_idx,

            named_params,

            params_type,

            global_other_params,

            shared_params_for_simulation

        )





    # ======================================================
    # Model parameter interface
    # ======================================================

    def set_model_params(
            self,
            model_parameters):


        self.trainer.set_model_params(
            model_parameters
        )



    def get_model_params(self):

        return self.trainer.get_model_params()



    def test(
            self,
            epoch):

        return self.trainer.test(

            epoch,

            self.test_dataloader,

            self.device

        )





    # ======================================================
    # abstract training function
    # implemented by FedAvg/FedNova client
    # ======================================================

    @abstractmethod
    def algorithm_on_train(
            self,
            share_data1,
            share_data2,
            share_y,
            round_idx,
            named_params,
            params_type="model",
            global_other_params=None,
            shared_params_for_simulation=None):

        raise NotImplementedError
