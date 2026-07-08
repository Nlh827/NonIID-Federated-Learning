# A Federated Feature Optimization and Personalization Framework for Non-IID Federated Learning

This repository provides an implementation of a federated learning framework designed to address the challenges of Non-Independent and Identically Distributed (Non-IID) data.

The proposed framework aims to improve communication efficiency and personalization capability under heterogeneous data distributions by introducing feature optimization and personalization strategies.

The framework integrates:

* **VAE-based proxy data generation** for proxy-based knowledge sharing without direct raw data exchange
* **Entropy-guided proxy sample selection** for improving proxy data quality
* **FedAvg-based local optimization**
* **FedSelect-based partial model aggregation** for communication-efficient personalization
* **Local personalization fine-tuning** for client-specific adaptation


## Overview

Federated learning enables collaborative model training among multiple clients without directly sharing raw client data. However, practical federated learning systems still face several challenges:

1. **Severe Non-IID data distributions**
2. **High communication costs caused by full model exchange**
3. **Limited adaptability of globally shared models for personalized clients**
4. **Difficulty in sharing informative knowledge without directly exposing raw data**

To address these challenges, this repository implements a unified federated learning framework consisting of the following components:

```
Communication Round
│
├── 1. VAE Proxy Data Generation
│       ├── Client-side VAE training
│       ├── VAE aggregation
│       └── Proxy data generation
│
├── 2. Proxy Data Optimization
│       ├── Entropy-based sample selection
│       └── Mixed local/proxy dataloader construction
│
├── 3. Federated Optimization
│       ├── Local client training
│       └── FedAvg model update
│
├── 4. FedSelect Personalization
│       ├── Layer-wise mask update
│       ├── Partial model upload
│       └── Global/local parameter fusion
│
└── 5. Personalization Stage
        └── Local fine-tuning
```


# Main Components


## 1. VAE-based Proxy Data Generation

Instead of directly sharing raw client samples, each client trains a variational autoencoder (VAE) to generate proxy representations.

The generated proxy data are aggregated and shared among clients to facilitate global knowledge transfer without direct raw data exchange.

Supported proxy modes:

```
noisy_residual
residual
reconstruction
raw
```


## 2. Entropy-guided Proxy Selection

To improve the quality of proxy data, this framework selects informative proxy samples according to model prediction uncertainty.

Available strategies:

```yaml
high_entropy
low_entropy
random
class_balanced_entropy
none
```

Example:

```yaml
entropy_selection_strategy: high_entropy
entropy_select_ratio: 0.6
```


## 3. FedAvg Training

The framework maintains the standard federated averaging optimization:

\[
w^{t+1}=\sum_k\frac{n_k}{n}w_k^t
\]


Each client optimizes the model using:

* local training samples
* selected proxy samples


## 4. FedSelect Communication-efficient Personalization

FedSelect introduces layer-wise parameter selection to achieve communication-efficient personalization.

Each parameter tensor is assigned a binary mask:

```
mask = 0 : global/shared parameter

mask = 1 : local/personalized parameter
```


Only global parameters are uploaded:

```
Client Model
     |
     |
Mask Selection
     |
     +---- Global parameters
     |          |
     |          v
     |      Server Aggregation
     |
     +---- Local parameters
                |
                v
          Client Preservation
```


This strategy reduces communication overhead while maintaining client-specific model adaptation.


## 5. Personalization Fine-tuning

After federated training, each client performs local adaptation using its own data.

The personalization stage starts after:

```yaml
local_personalize_start_frac: 0.8
```

of the total communication rounds.


# Supported Datasets

The current implementation supports:

* SVHN
* CIFAR-10
* CIFAR-100


# Requirements

Recommended environment:

```
Python >= 3.8

PyTorch
torchvision
numpy
scipy
scikit-learn
yacs
tqdm
wandb
pandas
```


Install dependencies:

```bash
pip install -r requirements.txt
```


# Project Structure

```
Framework/
│
├── main.py
│
├── algorithms/
│   └── basePS/
│
├── algorithms_standalone/
│   ├── fedavg/
│   └── basePS/
│
├── model/
│   ├── FL_VAE.py
│   └── cv/
│
├── trainers/
│
├── data_preprocessing/
│
├── configs/
│
├── optim/
│
├── loss_fn/
│
├── lr_scheduler/
│
└── utils/
```


# Configuration

Main configuration file:

```
configs/default.py
```


## Federated Setting

```python
client_num_in_total = 10
client_num_per_round = 5
comm_round = 300
partition_method = "hetero"
partition_alpha = 0.1
```


## FedSelect

```python
fedselect = True

lth_epoch_iters = 5

prune_target = 50
```


## VAE Proxy Data

```python
VAE = True

VAE_comm_round = 15

proxy_mode = "noisy_residual"

entropy_selection_strategy = "high_entropy"
```


## Personalization

```python
local_personalize = True

local_personalize_start_frac = 0.8
```


# Running

Example:

```bash
python main.py
```


For customized experiments:

```bash
python main.py \
--config_file configs/default.py
```


# Logging and Evaluation

The framework supports:

* Round-wise accuracy recording
* AUC convergence evaluation
* Communication cost statistics
* Training time measurement
* GPU memory statistics


Generated results:

```
results_revision/

├── round_logs.csv

├── summary.json

└── checkpoints/
```


# Privacy Statement

The proxy data generation mechanism is designed to reduce direct exposure of raw client data during federated knowledge sharing.

However, the current implementation **does not provide formal differential privacy guarantees**.

A rigorous privacy analysis, including:

* privacy budget estimation
* differential privacy bounds
* reconstruction attack evaluation

is considered future work.


# Acknowledgement

This project is developed based on the open-source FedFed framework:

> Zhiqin Yang, Yonggang Zhang, Yu Zheng, Xinmei Tian, Hao Peng, Tongliang Liu, Bo Han.
> FedFed: Feature Distillation against Data Heterogeneity in Federated Learning.
> NeurIPS 2023.

We sincerely thank the authors for making their code publicly available.

# License

This project is released under the MIT License.

This repository contains modifications and extensions based on the FedFed framework, which is also released under the MIT License.