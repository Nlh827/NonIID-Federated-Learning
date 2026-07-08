from .config import CfgNode as CN


_C = CN()


# ---------------------------------------------------------------------------- #
# Personalization
# ---------------------------------------------------------------------------- #

_C.local_personalize = True

_C.local_personalize_start_frac = 0.8

_C.local_personalize_val_frac = 0.2

_C.local_personalize_lr = 5e-5

_C.local_personalize_steps = 10

_C.local_personalize_init_rho = 0.3

_C.local_personalize_head_reg_lambda = 1e-3

_C.local_personalize_beta_candidates = [
    0.0,
    0.1,
    0.2,
    0.3
]

_C.local_personalize_accept_margin = 0.001



# ---------------------------------------------------------------------------- #
# FedSelect
# ---------------------------------------------------------------------------- #

_C.fedselect = True


_C.lth_epoch_iters = 5


_C.prune_percent = 5.0

# local parameter byte ratio
_C.prune_target = 50.0

# ["all_global", "all_local", "random"]
_C.mask_init = "all_global"



# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #

_C.dataset = "cifa10"

_C.client_num_in_total = 10

_C.client_num_per_round = 5

_C.gpu_index = 0

_C.num_classes = 10

_C.data_dir = "./../data"

_C.partition_method = "hetero"

_C.partition_alpha = 0.1



# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #

_C.model = "resnet18_v2"

_C.model_input_channels = 3

_C.model_output_dim = 10



# ---------------------------------------------------------------------------- #
# Algorithm
# ---------------------------------------------------------------------------- #

_C.algorithm = "FedAvg"



# ---------------------------------------------------------------------------- #
# Federated training
# ---------------------------------------------------------------------------- #

_C.global_epochs_per_round = 1

_C.comm_round = 300


_C.lr = 0.01

_C.seed = 0



# ---------------------------------------------------------------------------- #
# Revision experiment settings
# ---------------------------------------------------------------------------- #

_C.seed_list = [
    0,
    1
]


_C.exp_name = "entropy_SVHN_alpha0.1_ratio0.6"


_C.save_result_dir = "./results_revision"



# save options

_C.save_round_logs = True

_C.save_client_logs = True

_C.save_config = True


_C.save_final_model = True

_C.save_vae_model = True



# communication / cost

_C.enable_cost_log = True

_C.report_effective_comm = True



# evaluation

_C.target_acc_mode = "fixed"

_C.target_acc_value = 70.0

_C.target_acc_ratio = 0.95


_C.enable_auc = True

_C.test_every_round = True

_C.eval_personalized = True




# ---------------------------------------------------------------------------- #
# Entropy proxy selection
# ---------------------------------------------------------------------------- #

# high_entropy
# low_entropy
# random
# class_balanced_entropy
# none

_C.entropy_selection_strategy = "high_entropy"

_C.entropy_select_ratio = 0.6

_C.entropy_batch_size = 128



# ---------------------------------------------------------------------------- #
# Proxy data / VAE feature mode
# ---------------------------------------------------------------------------- #

# noisy_residual
# residual
# reconstruction
# raw

_C.proxy_mode = "noisy_residual"


_C.use_proxy_data = True



# ---------------------------------------------------------------------------- #
# Privacy enhancement
# ---------------------------------------------------------------------------- #

_C.privacy_clip = False

_C.privacy_clip_norm = 1.0


_C.privacy_noise_std = 0.01



# ---------------------------------------------------------------------------- #
# Ablation
# ---------------------------------------------------------------------------- #

_C.disable_feature_distill = False

_C.disable_entropy_selection = False

_C.disable_noise = False

_C.disable_partial_or_fedselect = False

_C.disable_personalization = False



# ---------------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------------- #

_C.record_tool = "wandb"

_C.wandb_record = False



# ---------------------------------------------------------------------------- #
# Data batch
# ---------------------------------------------------------------------------- #

_C.batch_size = 64

_C.VAE_batch_size = 64

_C.VAE_aug_batch_size = 64



# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #

_C.loss_fn = "CrossEntropy"

_C.exchange_model = True

# ---------------------------------------------------------------------------- #
# VAE settings
# ---------------------------------------------------------------------------- #

_C.VAE = True

_C.VAE_local_epoch = 1


# VAE latent dimension

_C.VAE_d = 32

_C.VAE_z = 2048



# VAE training scheduler

_C.VAE_sched = "cosine"

_C.VAE_sched_lr_ate_min = 2.e-3


_C.VAE_step = "+"



# VAE data augmentation

_C.VAE_mixupdata = False



# Reconstruction curriculum

_C.VAE_curriculum = True



# VAE noise

_C.VAE_mean = 0


_C.VAE_std1 = 0.2

_C.VAE_std2 = 0.25


_C.noise_type = "Gaussian"
# Gaussian or Laplace



# VAE loss weights

_C.VAE_re = 5.0

_C.VAE_ce = 10.0

_C.VAE_kl = 0.005

_C.VAE_x_ce = 0.4



# VAE federated rounds

_C.VAE_comm_round = 15

_C.VAE_client_num_per_round = 10


_C.VAE_adaptive = True



# ---------------------------------------------------------------------------- #
# Mode settings
# ---------------------------------------------------------------------------- #

_C.mode = "standalone"

_C.test = True

_C.instantiate_all = True


_C.client_index = 0



# ---------------------------------------------------------------------------- #
# Task
# ---------------------------------------------------------------------------- #

_C.task = "classification"



# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #

_C.dataset_aug = "default"

_C.dataset_resize = False

_C.dataset_load_image_size = 32


_C.data_efficient_load = True


# Dirichlet

_C.dirichlet_min_p = None

_C.dirichlet_balance = False



_C.data_load_num_workers = 0



# ---------------------------------------------------------------------------- #
# Data sampler
# ---------------------------------------------------------------------------- #

_C.data_sampler = "random"


_C.TwoCropTransform = False



# ---------------------------------------------------------------------------- #
# Feature settings
# ---------------------------------------------------------------------------- #

_C.model_out_feature = False

_C.model_out_feature_layer = "last"

_C.model_feature_dim = 512



_C.pretrained = False

_C.pretrained_dir = ""



# ---------------------------------------------------------------------------- #
# Generator / image
# ---------------------------------------------------------------------------- #

_C.image_resolution = 32



# ---------------------------------------------------------------------------- #
# Client selection
# ---------------------------------------------------------------------------- #

_C.client_select = "random"



# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #

_C.max_epochs = 90


_C.client_optimizer = "no"

_C.server_optimizer = "no"



_C.wd = 0.0001

_C.momentum = 0.9

_C.nesterov = False



# ---------------------------------------------------------------------------- #
# Learning rate scheduler
# ---------------------------------------------------------------------------- #

_C.sched = "no"

# no / StepLR / MultiStepLR / CosineAnnealingLR


_C.lr_decay_rate = 0.992


_C.step_size = 1


_C.lr_milestones = [
    30,
    60
]


_C.lr_T_max = 10


_C.lr_eta_min = 0


_C.lr_warmup_type = "constant"


_C.warmup_epochs = 0


_C.lr_warmup_value = 0.1



# ---------------------------------------------------------------------------- #
# Logging
# ---------------------------------------------------------------------------- #

_C.level = "INFO"



# ---------------------------------------------------------------------------- #
# Compatibility parameters
# ---------------------------------------------------------------------------- #

# result directory
_C.result_dir = "./results_revision"


# used by some trainers
_C.role = "client"

_C.server_index = 0



# ---------------------------------------------------------------------------- #
# Return config
# ---------------------------------------------------------------------------- #

def get_cfg_defaults():

    return _C.clone()