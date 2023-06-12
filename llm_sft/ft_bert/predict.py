# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: 推理


import traceback
import random
import time
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from llm_sft.ft_bert.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

# from peft import (LoraConfig, get_peft_model, prepare_model_for_int8_training)
from transformers import BertTokenizer, BertTokenizerFast
from transformers import BertLMHeadModel, BertConfig
from transformers import GenerationConfig
import torch

from llm_sft.ft_bert.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from llm_sft.ft_bert.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from llm_sft.ft_bert.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE
from llm_sft.ft_bert.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from llm_sft.ft_bert.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from llm_sft.ft_bert.config import USE_CUDA

# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
device_map = "auto"
# USE_CUDA = True
print(device_map)
print(ddp)


def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.bin"):
    """  仅保存 有梯度 的 模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        config.save_pretrained(model_save_dir)
        # config.to_dict()
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def print_named_parameters(model, use_print_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_data:
            print((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def generate_prompt(data_point, is_logger=False):
    """   模板   """
    if data_point["input"]:
        text_1, text_2 = f"""下面是一条指令。请根据问题，并编写一个准确的回答，以适当地完成指令。
        \n###指令：\n{data_point["instruction"]}
        \n###问题：\n{data_point["input"]}
        \n###回答：\n""", f"""{data_point["output"]}"""
    else:
        text_1, text_2 = f"""下面是一条指令。请编写一个准确的回答，以适当地完成指令。
        \n###指令：\n{data_point["instruction"]}
        \n###回答：\n""", f"""{data_point["output"]}"""

    x = tokenizer.encode(text_1.replace(" ", ""))[:-1]
    y = tokenizer.encode(text_2.replace(" ", ""))[1:-1]
    if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
        x = x[:MAX_LENGTH_Q]
        y = y[:MAX_LENGTH_A]
    if not x:
        y = [ID_PAD, ID_BOS]
    if x[-1] != ID_BOS:
        x += [ID_BOS]
    if not y:
        y = [ID_PAD, ID_EOS]
    if y and y[-1] != ID_EOS:
        y += [ID_EOS]
    out = {"input_ids": x, "labels": y}
    if is_logger:
        print(text_1)
        print(text_2)
        print(out)
        print(x)
        print(y)
    return out
def data_collator(batch):
    # there's probably a way to do this with the tokenizer settings
    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels"))
                    for i in range(len(batch))]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)
        labels = [-100] * len(x) + y + [-100] * len_padding
        input_ids = x + y + [ID_PAD] * len_padding
        attention_mask = [0] * len(x) + [1] * (len_max_batch-len(x))
        tensor_attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_attention_mask.append(tensor_attention_mask)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = {"attention_mask": batch_attention_mask,
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    return input_dict


model = BertLMHeadModel.from_pretrained(PATH_MODEL_PRETRAIN)
# model.gradient_checkpointing_disable()
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
# model.is_parallelizable = False
# model.config.use_cache = False
# model.model_parallel = False
print("load BertGenerationDecoder ok")

print_named_parameters(model)
tokenizer = BertTokenizer.from_pretrained(PATH_MODEL_PRETRAIN, add_eos_token=True)
ID_MASK = tokenizer.convert_tokens_to_ids("[MASK]")
ID_PAD = tokenizer.convert_tokens_to_ids("[PAD]")
ID_BOS = tokenizer.convert_tokens_to_ids("<S>")
ID_EOS = tokenizer.convert_tokens_to_ids("<T>")
ID_CLS = tokenizer.convert_tokens_to_ids("[CLS]")
ID_SEP = tokenizer.convert_tokens_to_ids("[SEP]")
print(ID_MASK)
print(ID_PAD)
print(ID_BOS)
print(ID_EOS)
print(ID_CLS)
print(ID_SEP)
### load-pretrained
path_model_checkpoint = os.path.join(MODEL_SAVE_DIR, "adapter_model.bin")
state_dict_checkpoint = torch.load(path_model_checkpoint, map_location=torch.device("cpu"))
model.load_state_dict(state_dict_checkpoint, strict=False)
del state_dict_checkpoint

if USE_CUDA:
    model = model.cuda()
else:
    model = model.bfloat16()
print_named_parameters(model, True)


def predict(data_dict):
    """  推理  """
    prompt_dict = generate_prompt(data_dict, is_logger=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        num_beams=1,
        do_sample=True,
        penalty_alpha=1.05,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            # max_new_tokens=502,
            # max_new_tokens=MAX_LENGTH_QA, # 使用max_new_tokens有时候会超过512
            max_length=MAX_LENGTH_QA,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(prompt_dict)
    print(input_ids)
    print(output)
    # output = output.split("答：")[-1]
    return output


if __name__ == '__main__':
    data_dict = {"instruction": "解释为什么下面的分数等于 1/4",
                 "input": "解释为什么下面的分数等于 1/4，4/16",
                 "output": "分数 4/16 等于 1/4，因为分子和分母都可以被 4 整除。将顶部和底部数字都除以 4 得到分数 1/4。"
                 }
    res = predict(data_dict)
    print(res)
    while True:
        try:
            time_start = time.time()
            history = []
            print("请输入:")
            ques = input()
            print("请稍等...")

            if ques.strip().upper() == "CLEAR":
                history = []
                print("clear ok")
                continue
            else:
                ques_dict = {"instruction": ques, "input": "", "output": ""}
                # ques_dict = ques
                res = predict(ques_dict)
                print(res)
            print(time.time() - time_start)
        except Exception as e:
            print(traceback.print_exc())
            print(str(e))


"""
预测, predict
"""

