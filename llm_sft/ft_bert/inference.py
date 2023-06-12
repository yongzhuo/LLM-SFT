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


class BertSFT:
    def __init__(self, PATH_MODEL_PRETRAIN, MODEL_SAVE_DIR, USE_CUDA):
        self.PATH_MODEL_PRETRAIN = PATH_MODEL_PRETRAIN
        self.MODEL_SAVE_DIR = MODEL_SAVE_DIR
        self.USE_CUDA = USE_CUDA
        self._load_model()

    def _load_model(self):
        self.model = BertLMHeadModel.from_pretrained(self.PATH_MODEL_PRETRAIN)
        # model.gradient_checkpointing_disable()
        # model.gradient_checkpointing_enable()
        # model.enable_input_require_grads()
        # model.is_parallelizable = False
        # model.config.use_cache = False
        # model.model_parallel = False
        print("load BertGenerationDecoder ok")

        print_named_parameters(self.model)
        self.tokenizer = BertTokenizer.from_pretrained(self.PATH_MODEL_PRETRAIN, add_eos_token=True)
        self.ID_MASK = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.ID_PAD = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.ID_BOS = self.tokenizer.convert_tokens_to_ids("<S>")
        self.ID_EOS = self.tokenizer.convert_tokens_to_ids("<T>")
        self.ID_CLS = self.tokenizer.convert_tokens_to_ids("[CLS]")
        self.ID_SEP = self.tokenizer.convert_tokens_to_ids("[SEP]")
        print(self.ID_MASK)
        print(self.ID_PAD)
        print(self.ID_BOS)
        print(self.ID_EOS)
        print(self.ID_CLS)
        print(self.ID_SEP)
        ### load-pretrained
        path_model_checkpoint = os.path.join(self.MODEL_SAVE_DIR, "adapter_model.bin")
        state_dict_checkpoint = torch.load(path_model_checkpoint, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict_checkpoint, strict=False)
        del state_dict_checkpoint

        if self.USE_CUDA:
            model = self.model.cuda()
        else:
            model = self.model.bfloat16()
        print_named_parameters(model, True)

    def generate_prompt(self, data_point, is_logger=False):
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

        x = self.tokenizer.encode(text_1.replace(" ", ""))[:-1]
        y = self.tokenizer.encode(text_2.replace(" ", ""))[1:-1]
        if len(x) + len(y) > (MAX_LENGTH_Q + MAX_LENGTH_A):
            x = x[:MAX_LENGTH_Q]
            y = y[:MAX_LENGTH_A]
        if not x:
            y = [self.ID_PAD, self.ID_BOS]
        if x[-1] != self.ID_BOS:
            x += [self.ID_BOS]
        if not y:
            y = [self.ID_PAD, self.ID_EOS]
        if y and y[-1] != self.ID_EOS:
            y += [self.ID_EOS]
        out = {"input_ids": x, "labels": y}
        if is_logger:
            print(text_1)
            print(text_2)
            print(out)
            print(x)
            print(y)
        return out

    def predict(self, data_dict):
        """  推理  """
        prompt_dict = self.generate_prompt(data_dict, is_logger=True)
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
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                # max_new_tokens=502,
                # max_new_tokens=MAX_LENGTH_QA, # 使用max_new_tokens有时候会超过512
                max_length=MAX_LENGTH_QA,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True,)
        print(prompt_dict)
        print(input_ids)
        print(output)
        # output = output.split("答：")[-1]
        return output

    def chat(self, query, history=None, max_len=510):
        """   chat, 聊天   """
        prompt = ""
        if history:
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

            if len(prompt) + len(query) > max_len:
                query_new = "[Round {}]\n问：{}\n答：".format(len(history), query)
                prompt = prompt[:int(max_len/2)] + query_new[:int(max_len/2)]
            else:
                prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        else:
            prompt = query
        data_dict = {"instruction": prompt, "input": "", "output": ""}
        out = self.predict(data_dict)
        out = out.split("答 ：")[-1]
        if not history:
            history = [(query, out)]
        else:
            history += [(query, out)]
        return out, history


if __name__ == '__main__':
    from llm_sft.ft_bert.config import PATH_MODEL_PRETRAIN, MODEL_SAVE_DIR, REPO_ID
    bert_sft = BertSFT(PATH_MODEL_PRETRAIN=PATH_MODEL_PRETRAIN,
                       MODEL_SAVE_DIR=MODEL_SAVE_DIR,
                       USE_CUDA=True)
    text = "1+1="
    data_dict = {"instruction": text, "input": "", "output": ""}
    res = bert_sft.predict(data_dict)
    print(res)

