# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: 验证评估


import logging as logger
import traceback
import random
import json
import time
import sys
import os
sys.setrecursionlimit(100000)
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, modified_precision
# from peft import (LoraConfig, get_peft_model, prepare_model_for_int8_training)
from transformers import BertGenerationDecoder, BertConfig
from transformers import BertTokenizer, BertTokenizerFast
from transformers import GenerationConfig
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import torch

from llm_sft.ft_bert.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from llm_sft.ft_bert.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from llm_sft.ft_bert.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE
from llm_sft.ft_bert.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from llm_sft.ft_bert.config import LORA_DROPOUT, LORA_ALPHA, LORA_R
from llm_sft.ft_bert.config import USE_CUDA


# USE_CUDA = True
device_map = "auto"
# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
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
def txt_read(path, encode_type="utf-8", errors=None):
    """
        读取txt文件，默认utf8格式, 不能有空行
    Args:
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
        errors[String]: specifies how encoding errors handled, eg. "ignore", strict
    Returns:
        lines[List]: output lines
    """
    lines = []
    try:
        file = open(path, "r", encoding=encode_type, errors=errors)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines
def load_json(path: str, encoding: str="utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json
def generate_prompt(data_point, is_logger=False):
    # sorry about the formatting disaster gotta move fast
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
        # labels = [-100] * len_padding + [-100] * len(x) + y
        # input_ids = [ID_PAD] * len_padding + x + y
        # attention_mask = [1] * (len_max_batch - len(x)) + [0] * len(x)
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


model = BertGenerationDecoder.from_pretrained(PATH_MODEL_PRETRAIN)
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

### load-pretrained
path_model_checkpoint = os.path.join(MODEL_SAVE_DIR, "adapter_model.bin")
state_dict_checkpoint = torch.load(path_model_checkpoint, map_location=torch.device("cpu"))
model.load_state_dict(state_dict_checkpoint, strict=False)
del state_dict_checkpoint

if USE_CUDA:
    # model = model.half().cuda()
    model = model.cuda()
else:
    model = model.bfloat16()
# model = model.bfloat16()
print_named_parameters(model, True)


def predict(data_dict, generation_dict):
    """  推理  """
    xy_dict = generate_prompt(data_dict, is_logger=True)
    input_ids = xy_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(**generation_dict)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=MAX_LENGTH_QA,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(xy_dict)
    print(input_ids)
    print(output)
    # output = output.split("答：")[-1]
    return output,


def text_generate(request_data):
    instruction = request_data.instruction
    text = request_data.text
    max_new_tokens = request_data.max_new_tokens
    penalty_alpha = request_data.penalty_alpha
    temperature = request_data.temperature
    do_sample = request_data.do_sample
    num_beams = request_data.num_beams
    top_p = request_data.top_p
    top_k = request_data.top_k

    generation_dict = vars(request_data)
    print(generation_dict)
    generation_dict.pop("max_new_tokens")
    generation_dict.pop("instruction")
    generation_dict.pop("text")
    data_point = {"instruction": instruction, "input": text, "output": ""}
    generation_dict = {"temperature": temperature,
                         "top_p": top_p,
                         "top_k": top_k,
                         "num_beams": num_beams,
                         "do_sample": do_sample,
                         "penalty_alpha": penalty_alpha,
                         }

    try:  # 数据预处理, 模型预测
        response = predict(data_point, generation_dict)
    except Exception as e:
        logger.info(traceback.print_exc())
        response = "[SEP]"
    return response
class Item(BaseModel):
    instruction: str = "完成下面的对话。"
    text: str = ""
    penalty_alpha: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 0.95  # 0.95  # 0.35  # 0.95
    do_sample: bool = True
    num_beams: int = 1
    top_p: float = 0.7
    top_k: int = 50


if __name__ == '__main__':
    text = "1+1="
    item_config = Item()
    item_config.text = text
    response = text_generate(item_config)
    print(response)

    smooth = SmoothingFunction().method1
    rouge = Rouge()
    best_bleu = 0.


    fw = open(DATA_PATH + ".bert_unilm_eval_rouge_blue.512.json", "a+", encoding="utf-8")
    rouge_1_p, rouge_2_p, rouge_l_p = 0, 0, 0
    rouge_1_r, rouge_2_r, rouge_l_r = 0, 0, 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    total = 0
    time_start = time.time()
    datas = load_json(DATA_PATH)
    # datas = datas[:1024]
    datas = datas[:8]
    for d_json in tqdm(datas, desc="data"):
        try:
            instruction = d_json.get("instruction", "")
            text_input = d_json.get("input", "")
            text_output = d_json.get("output", "")
            # qtext, qans
            total += 1
            item_config = Item()
            item_config.instruction = instruction
            item_config.text = text_input
            text_output = " ".join(list(text_output))
            text_pred = " ".join(list(text_generate(item_config))).lower() \
                .replace(" ", "").replace("[sep]", "")
            text_pred = " ".join(list(text_pred))
            line = {"input": text_input, "output": text_output.replace(" ", ""),
                    "pred": text_pred.replace(" ", "")}
            line_str = json.dumps(line, ensure_ascii=False) + "\n"
            fw.write(line_str)
            if text_pred.strip():
                scores = rouge.get_scores(hyps=text_pred, refs=text_output)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']

                rouge_1_p += scores[0]['rouge-1']['p']
                rouge_2_p += scores[0]['rouge-2']['p']
                rouge_l_p += scores[0]['rouge-l']['p']
                rouge_1_r += scores[0]['rouge-1']['r']
                rouge_2_r += scores[0]['rouge-2']['r']
                rouge_l_r += scores[0]['rouge-l']['r']

                bleu += sentence_bleu(references=[list(text_output)],
                                      hypothesis=list(text_pred),
                                      smoothing_function=smooth)
                mertics_i = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l,
                             'bleu': bleu,
                             'rouge-1_p': rouge_1_p, 'rouge-2_p': rouge_2_p, 'rouge-l_p': rouge_l_p,
                             'rouge-1_r': rouge_1_r, 'rouge-2_r': rouge_2_r, 'rouge-l_r': rouge_l_r, }
                if total < 5:
                    print(text_output.replace(" ", ""))
                    print(text_pred.replace(" ", ""))
                    print(mertics_i)
        except Exception as e:
            print(traceback.print_exc())
            continue
    time_end = time.time()
    lost_time = time_end - time_start
    lost_time_avg = lost_time / total
    rouge_1, rouge_2, rouge_l, bleu = rouge_1 / total, rouge_2 / total, rouge_l / total, bleu / total
    rouge_1_p, rouge_2_p, rouge_l_p = rouge_1_p / total, rouge_2_p / total, rouge_l_p / total
    rouge_1_r, rouge_2_r, rouge_l_r = rouge_1_r / total, rouge_2_r / total, rouge_l_r / total

    mertics = {'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l, 'bleu': bleu,
               "lost_time": lost_time, "lost_time_avg": lost_time_avg,
               'rouge-1_p': rouge_1_p, 'rouge-2_p': rouge_2_p, 'rouge-l_p': rouge_l_p,
               'rouge-1_r': rouge_1_r, 'rouge-2_r': rouge_2_r, 'rouge-l_r': rouge_l_r,
               }
    mertics = {k: round(v, 4) for k, v in mertics.items()}
    print(mertics, lost_time, lost_time_avg)
    fw.write(json.dumps(mertics, ensure_ascii=False) + "\n")
    fw.close()


"""
测评eval 
"""

