# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:29
# @author  : Mo
# @function: config of chatglm


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 2  # default=4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 5e-5  # default=3e-4  # the Karpathy constant
EPOCHS = 3  # default=3  # we don't always need 3 tbh

# LORA_DROPOUT = 0.1
# LORA_ALPHA = 32
# LORA_R = 32
LORA_DROPOUT = 0.05
LORA_ALPHA = 16
LORA_R = 8
SAVE_STEPS = 382
VAL_SET_SIZE = 0
MAX_LENGTH_Q = 256 - 2  # default=128 - 2
MAX_LENGTH_A = 256 - 2  # default=128 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2
TARGET_MODULES = ["query_key_value"]

PATH_MODEL_PRETRAIN = ""
REPO_ID = "THUDM/chatglm-6b"
PATH_MODEL_PRETRAIN = PATH_MODEL_PRETRAIN if PATH_MODEL_PRETRAIN else REPO_ID
DATA_PATH = "../dataset/alpaca_gpt4_data_zh.json"
MODEL_SAVE_DIR = "model_sft"

IS_PARALLELIZABLE = False
MODEL_PARALLEL = False
USE_CACHE = False
CUDA_VISIBLE_DEVICES = "0"
USE_TORCH = "1"
CPU_NUMS = "9"
USE_CUDA = False if CUDA_VISIBLE_DEVICES == "-1" else True


WARMUP_NUM_STEPS = 40
STEPS_PER_PRINT = 20
MAX_GRAD_NORM = 1.0

DEEPSPEED_CONF_2 = {
    # "fp16": {
    #     "enabled": True,
    #     # 0为自动, 在训练过程中，如果不发生上溢出，在更新scale_window次参数后，会尝试扩大scale的值；如果发生了上溢出，则跳过参数更新，并缩小scale的值，入参scale_factor是控制扩大或缩小的步数，scale_window控制没有发生溢出时，最大的连续更新步数。
    #     "loss_scale": 0,
    #     "loss_scale_window": 500,
    #     # 500 # 当overflow为False时最大连续正常步数。连续scale_window个step没有发生上溢出，则执行init_loss_scale=init_loss_scale*scale_factor
    #     "initial_scale_power": 16,  # init_loss_scale=2**24
    #     "hysteresis": 2,
    #     "min_loss_scale": 1
    # },
    #
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": LEARNING_RATE,
            "betas": [0.9, 0.999],
            "eps": 1e-5,
            "weight_decay": 5e-4,
        }
    },
    #
    # "scheduler": {
    #     "type": "WarmupLR",
    #     "params": {
    #         "warmup_min_lr": 0,
    #         "warmup_max_lr": LEARNING_RATE,
    #         "warmup_num_steps": WARMUP_NUM_STEPS,
    #     }
    # },

    "zero_optimization": {
        "stage": 2,
        # 可以看到，除了和stage2一样，有offload_optimizer参数之外，stage3还有一个offload_param参数。即，将模型参数进行划分。
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        # 另外一个需要提到的参数是overlap_comm。简单地理解，它控制着多个memory上进程之间通信的buffer的大小。
        # 这个值越大，进程之间通信越快，模型训练速度也会提升，
        # 但相应的显存占用也会变大；反之亦然。因此，overlap_comm也是一个需要进行一定权衡的参数。
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": int(BATCH_SIZE//MICRO_BATCH_SIZE),
    "gradient_clipping": MAX_GRAD_NORM,
    "train_batch_size": BATCH_SIZE,
    "steps_per_print": STEPS_PER_PRINT,
    "wall_clock_breakdown": False
}
DEEPSPEED_CONF_3 = {
    # "bfloat16": {
    #     "enabled": False
    # },
    # "fp16": {
    #     "enabled": True,
    #     "loss_scale": 0,
    #     "loss_scale_window": 1000,
    #     "initial_scale_power": 16,
    #     "hysteresis": 2,
    #     "min_loss_scale": 1
    # },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": LEARNING_RATE,
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        }
    },
    # "scheduler": {
    #     "type": "WarmupLR",
    #     "params": {
    #         "warmup_min_lr": 0,
    #         "warmup_max_lr": 2e-05,
    #         "warmup_num_steps": 10
    #     }
    # },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": False
        },
        # "overlap_comm": True,
        # "contiguous_gradients": True,
        # "sub_group_size": 1e9,
        # "reduce_bucket_size": "auto",

        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8,
        "sub_group_size": 1e9,
        "reduce_scatter": True,
        "overlap_comm": True,
        "contiguous_gradients": True,

        "stage3_prefetch_bucket_size": 1e9,
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_fp16_weights_on_model_save": True
    },
    "gradient_accumulation_steps": 8,
    "train_micro_batch_size_per_gpu": 4,
    "train_batch_size": 32,
    "steps_per_print": 20,
    "allgather_partitions": True,
    "allgather_bucket_size": 2e8,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": True
}
DEEPSPEED_CONF = DEEPSPEED_CONF_2

