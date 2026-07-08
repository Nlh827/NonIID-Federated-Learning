import copy
import logging

from .client import FedAVGClient
from .aggregator import FedAVGAggregator

from utils.data_utils import get_avg_num_iterations

from algorithms_standalone.basePS.basePSmanager import BasePSManager

from model.build import create_model
from trainers.build import create_trainer

from model.FL_VAE import FL_CVAE_cifar

from utils.fedselect_utils import merge_global_into_client


class FedAVGManager(BasePSManager):

    def __init__(self, device, args):

        super().__init__(
            device,
            args
        )

        self.global_epochs_per_round = (
            self.args.global_epochs_per_round
        )



    def _setup_server(self):

        logging.info(
            "############_setup_server (START)#############"
        )


        # ==========================
        # Global model
        # ==========================

        model = create_model(
            self.args,
            model_name=self.args.model,
            output_dim=self.args.model_output_dim,
            device=self.device,
            **self.other_params
        )


        init_state_kargs = {}


        # ==========================
        # VAE model
        # ==========================

        VAE_model = None

        if self.args.VAE:

            VAE_model = FL_CVAE_cifar(
                args=self.args,
                d=self.args.VAE_d,
                z=self.args.VAE_z,
                device=self.device
            )


        # ==========================
        # Trainer
        # ==========================

        model_trainer = create_trainer(
            self.args,
            self.device,
            model,

            train_data_global_num=self.train_data_global_num,
            test_data_global_num=self.test_data_global_num,

            train_data_global_dl=self.train_data_global_dl,
            test_data_global_dl=self.test_data_global_dl,

            train_data_local_num_dict=self.train_data_local_num_dict,

            class_num=self.class_num,

            server_index=0,
            role="server",

            **init_state_kargs
        )


        # ==========================
        # Aggregator
        # ==========================

        self.aggregator = FedAVGAggregator(
            self.train_data_global_dl,
            self.test_data_global_dl,

            self.train_data_global_num,
            self.test_data_global_num,

            self.train_data_local_num_dict,

            self.args.client_num_in_total,

            self.device,

            self.args,

            model_trainer,

            VAE_model
        )


        logging.info(
            "############_setup_server (END)#############"
        )



    def _setup_clients(self):

        logging.info(
            "############setup_clients (START)#############"
        )


        init_state_kargs = (
            self.get_init_state_kargs()
        )


        for client_index in range(
            self.number_instantiated_client
        ):


            # ==========================
            # Client VAE
            # ==========================

            VAE_model = None

            if self.args.VAE:

                VAE_model = FL_CVAE_cifar(
                    args=self.args,
                    d=self.args.VAE_d,
                    z=self.args.VAE_z,
                    device=self.device
                )



            # ==========================
            # Client model
            # ==========================

            model = create_model(
                self.args,

                model_name=self.args.model,

                output_dim=self.args.model_output_dim,

                device=self.device,

                **self.other_params
            )



            # ==========================
            # Trainer
            # ==========================

            model_trainer = create_trainer(
                self.args,
                self.device,
                model,

                class_num=self.class_num,

                other_params=self.other_params,

                client_index=client_index,

                role="client",

                **init_state_kargs
            )



            client = FedAVGClient(

                client_index,

                train_ori_data=
                self.train_data_local_ori_dict[client_index],

                train_ori_targets=
                self.train_targets_local_ori_dict[client_index],

                test_dataloader=
                self.test_data_local_dl_dict[client_index],

                train_data_num=
                self.train_data_local_num_dict[client_index],

                test_data_num=
                self.test_data_local_num_dict[client_index],

                train_cls_counts_dict=
                self.train_cls_local_counts_dict[client_index],

                device=self.device,

                args=self.args,

                model_trainer=model_trainer,

                vae_model=VAE_model,

                dataset_num=self.train_data_global_num
            )


            self.client_list.append(
                client
            )


        logging.info(
            "############setup_clients (END)#############"
        )



    # override
    def check_end_epoch(self):

        return True

    def algorithm_train(
            self,
            round_idx,
            client_indexes,
            named_params,
            params_type,
            global_other_params,
            update_state_kargs,
            shared_params_for_simulation):


        use_fedselect = getattr(
            self.args,
            "fedselect",
            False
        )


        logging.info(
            f"[Round {round_idx}] Local training stage start"
        )


        # =====================================================
        # FedSelect mask initialization
        # =====================================================

        if use_fedselect:

            if "client_masks" not in global_other_params:

                logging.info(
                    f"[Round {round_idx}] Initialize FedSelect masks"
                )


                from utils.fedselect_utils import (
                    init_client_mask_layerwise
                )


                client_masks = {}


                for cid in range(
                    len(self.client_list)
                ):

                    client_masks[cid] = (
                        init_client_mask_layerwise(
                            self.client_list[cid].trainer.model,

                            mode=self.args.mask_init,

                            seed=
                            int(self.args.seed)
                            +
                            int(cid)
                        )
                    )


                global_other_params[
                    "client_masks"
                ] = client_masks


            else:

                client_masks = (
                    global_other_params[
                        "client_masks"
                    ]
                )



        # =====================================================
        # Client local training
        # =====================================================


        for i, client_index in enumerate(
            client_indexes
        ):


            copy_global_other_params = copy.deepcopy(
                global_other_params
            )


            if use_fedselect:

                copy_global_other_params[
                    "client_mask"
                ] = client_masks[
                    client_index
                ]



            if self.args.exchange_model:

                copy_named_model_params = (
                    copy.deepcopy(named_params)
                )

            else:

                copy_named_model_params = named_params



            if self.args.instantiate_all:

                client = self.client_list[
                    client_index
                ]

            else:

                client = self.client_list[i]



            (
                model_params,
                model_indexes,
                local_sample_number,
                client_other_params,

                shared_params_for_simulation

            ) = client.train(

                self.global_share_dataset1,

                self.global_share_dataset2,

                self.global_share_data_y,

                round_idx,

                copy_named_model_params,

                params_type,

                copy_global_other_params,

                shared_params_for_simulation=
                shared_params_for_simulation
            )



            # receive updated mask

            if (
                use_fedselect
                and
                "client_mask"
                in client_other_params
            ):

                client_masks[
                    client_index
                ] = client_other_params[
                    "client_mask"
                ]



            self.aggregator.add_local_trained_result(

                client_index,

                model_params,

                model_indexes,

                local_sample_number,

                client_other_params
            )



        logging.info(
            f"[Round {round_idx}] Local training stage finished"
        )



        # =====================================================
        # Aggregation
        # =====================================================


        (
            global_model_params,
            global_other_params,

            shared_params_for_simulation

        ) = self.aggregator.aggregate()



        params_type = "model"



        if use_fedselect:

            global_other_params[
                "client_masks"
            ] = client_masks



        logging.info(
            f"[Round {round_idx}] Aggregation finished"
        )



        # =====================================================
        # FedSelect global/local parameter merge
        # =====================================================


        logging.info(
            f"[Round {round_idx}] Model distribution stage"
        )


        if use_fedselect:


            for cid in range(
                len(self.client_list)
            ):


                # keep local parameters

                client_params = copy.deepcopy(
                    self.client_list[cid]
                    .trainer
                    .get_model_params()
                )


                merged = merge_global_into_client(

                    client_params,

                    global_model_params,

                    client_masks[cid]

                )


                self.client_list[cid].set_model_params(
                    merged
                )


        else:


            for cid in range(
                len(self.client_list)
            ):

                self.client_list[cid].set_model_params(
                    global_model_params
                )



        # =====================================================
        # Personalization stage
        # =====================================================


        if getattr(
            self.args,
            "local_personalize",
            False
        ):


            comm_round = getattr(
                self.args,
                "comm_round",
                None
            )


            assert (
                comm_round is not None
                and int(comm_round) > 0
            ), (
                f"Invalid comm_round: {comm_round}"
            )


            comm_round = int(
                comm_round
            )


            frac = float(
                getattr(
                    self.args,
                    "local_personalize_start_frac",
                    0.8
                )
            )


            start_round = int(
                comm_round * frac
            )


            if round_idx >= start_round:


                logging.info(
                    f"[Round {round_idx}] "
                    "Personalization stage start"
                )


                for client_index in client_indexes:

                    self.client_list[
                        client_index
                    ].personalize_after_round(
                        round_idx
                    )


            else:

                logging.debug(
                    f"[Round {round_idx}] "
                    f"Skip personalization, "
                    f"start at {start_round}"
                )



        # =====================================================
        # Client evaluation
        # =====================================================


        epoch_tag = (
            self.args.VAE_comm_round
            +
            round_idx
        )


        selected_accs = []


        for cid in client_indexes:

            acc = self.client_list[cid].test(
                epoch_tag
            )

            selected_accs.append(
                acc
            )



        avg_selected_acc = (

            sum(selected_accs)
            /
            len(selected_accs)

            if len(selected_accs) > 0

            else 0.0

        )



        if not hasattr(
            self,
            "client_test_acc_list"
        ):

            self.client_test_acc_list = []



        self.client_test_acc_list.append(
            avg_selected_acc
        )



        logging.info(
            f"[Round {round_idx}] "
            f"Selected client accs: {selected_accs}"
        )


        logging.info(
            f"[Round {round_idx}] "
            f"Selected avg acc: {avg_selected_acc:.4f}"
        )



        return (

            global_model_params,

            params_type,

            global_other_params,

            shared_params_for_simulation

        )