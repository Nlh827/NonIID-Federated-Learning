# utils/fedselect_utils.py

import random
import copy
import logging

import torch



# ============================================================
# Initialize client mask
# ============================================================

def init_client_mask_layerwise(
        model,
        mode="all_global",
        p_local=0.5,
        seed=None):

    """
    mask[name] == 0:
        global/shared parameter
        participate in aggregation

    mask[name] == 1:
        local/personalized parameter
        kept locally
    """

    if seed is not None:

        rng = random.Random(seed)

    else:

        rng = random



    mask = {}


    for name, param in model.named_parameters():


        if not param.requires_grad:

            continue



        if mode == "all_global":

            mask[name] = 0



        elif mode == "all_local":

            mask[name] = 1



        elif mode == "random":

            mask[name] = (
                1
                if rng.random() < p_local
                else 0
            )



        else:

            raise ValueError(
                f"Unknown mask init mode: {mode}"
            )



    return mask





# ============================================================
# Update mask according to local update magnitude
# ============================================================

def delta_update_mask_layerwise(
        old_params,
        new_params,
        mask,
        prune_percent=20.0,
        prune_target=80.0):


    """
    Byte-aware layer-wise FedSelect update.

    Large delta:
        more client-specific
        -> local

    Small delta:
        more shared
        -> global


    mask:
        0 : global
        1 : local
    """



    deltas = {}

    param_bytes = {}



    for name in mask.keys():


        if (
            name not in old_params
            or
            name not in new_params
        ):

            continue



        new_tensor = new_params[name]

        old_tensor = (
            old_params[name]
            .to(new_tensor.device)
        )



        if not torch.is_floating_point(
                new_tensor):

            continue



        deltas[name] = torch.norm(
            new_tensor - old_tensor
        ).item()



        param_bytes[name] = (
            new_tensor.numel()
            *
            new_tensor.element_size()
        )



    if len(deltas) == 0:

        return mask



    total_bytes = sum(
        param_bytes.values()
    )


    target_local_bytes = (
        total_bytes
        *
        float(prune_target)
        /
        100.0
    )



    max_change_bytes = (
        total_bytes
        *
        float(prune_percent)
        /
        100.0
    )



    current_local_bytes = sum(

        param_bytes[k]

        for k in param_bytes

        if mask.get(k, 0) == 1

    )



    high_delta = sorted(
        deltas.items(),
        key=lambda x:x[1],
        reverse=True
    )


    low_delta = sorted(
        deltas.items(),
        key=lambda x:x[1]
    )



    changed_bytes = 0



    # --------------------------------------------------------
    # Increase local parameters
    # --------------------------------------------------------

    if current_local_bytes < target_local_bytes:


        for name, _ in high_delta:


            if changed_bytes >= max_change_bytes:

                break



            if mask.get(name,0) == 0:


                mask[name] = 1


                changed_bytes += (
                    param_bytes[name]
                )


                current_local_bytes += (
                    param_bytes[name]
                )



            if current_local_bytes >= target_local_bytes:

                break



    # --------------------------------------------------------
    # Increase global parameters
    # --------------------------------------------------------

    elif current_local_bytes > target_local_bytes:


        for name, _ in low_delta:


            if changed_bytes >= max_change_bytes:

                break



            if mask.get(name,0) == 1:


                mask[name] = 0


                changed_bytes += (
                    param_bytes[name]
                )


                current_local_bytes -= (
                    param_bytes[name]
                )



            if current_local_bytes <= target_local_bytes:

                break



    return mask





# ============================================================
# Deprecated compatibility function
# ============================================================

def apply_mask_requires_grad_layerwise(
        model,
        mask):


    """
    FedSelect does NOT freeze parameters during training.

    All parameters participate in local optimization.

    This function is kept only for backward compatibility.
    """


    for name, param in model.named_parameters():

        if name in mask:

            param.requires_grad = True





# ============================================================
# Partial upload
# ============================================================

def select_global_params_for_upload(
        state_dict,
        mask):


    """
    Select parameters for communication.

    mask==0:
        upload

    mask==1:
        keep local


    Buffers:
        BN statistics are uploaded by default.
    """



    upload_state = {}

    full_bytes = 0

    upload_bytes = 0



    for name, tensor in state_dict.items():


        if not torch.is_tensor(tensor):

            continue



        tensor_cpu = (
            tensor.detach()
            .cpu()
        )



        size_bytes = (
            tensor_cpu.numel()
            *
            tensor_cpu.element_size()
        )



        full_bytes += size_bytes



        if name in mask:


            upload = (
                mask[name] == 0
            )


        else:


            upload = True



        if upload:


            upload_state[name] = tensor_cpu

            upload_bytes += size_bytes




    ratio = (

        upload_bytes / full_bytes

        if full_bytes > 0

        else 1.0

    )



    stats = {

        "full_model_mb":
            full_bytes
            /
            1024.0
            /
            1024.0,


        "uploaded_model_mb":
            upload_bytes
            /
            1024.0
            /
            1024.0,


        "upload_ratio":
            ratio,


        "num_full_tensors":
            len(state_dict),


        "num_uploaded_tensors":
            len(upload_state),


        "num_local_tensors":
            sum(
                1
                for v in mask.values()
                if v == 1
            ),


        "num_global_tensors":
            sum(
                1
                for v in mask.values()
                if v == 0
            ),

    }



    return upload_state, stats





# ============================================================
# Merge global model into client
# ============================================================

def merge_global_into_client(
        client_params,
        global_params,
        client_mask):


    """
    Synchronize global parameters.

    mask==0:
        replace with global value

    mask==1:
        keep client value
    """



    merged = copy.deepcopy(
        client_params
    )



    for name, value in global_params.items():


        if name in client_mask:


            if client_mask[name] == 0:

                merged[name] = value



        else:


            # BN buffer etc.

            merged[name] = value



    return merged