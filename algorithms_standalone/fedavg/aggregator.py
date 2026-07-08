# algorithms_standalone/fedavg/aggregator.py

import logging

from algorithms_standalone.basePS.aggregator import Aggregator
from model.build import create_model


class FedAVGAggregator(Aggregator):

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


        super().__init__(
            train_dataloader,
            test_dataloader,
            train_data_num,
            test_data_num,
            train_data_local_num_dict,
            worker_num,
            device,
            args,
            model_trainer,
            vae_model
        )



        # ==================================================
        # SCAFFOLD global control variate
        # ==================================================

        if getattr(
                self.args,
                "scaffold",
                False):


            self.c_model_global = create_model(
                self.args,
                model_name=self.args.model,
                output_dim=self.args.model_output_dim,
                device=self.device
            )


            for _, params in self.c_model_global.named_parameters():

                params.data.zero_()



            logging.info(
                "Initialized SCAFFOLD global control model"
            )



    # ==================================================
    # Communication rounds
    # ==================================================

    def get_max_comm_round(self):

        return int(
            self.args.comm_round
        )