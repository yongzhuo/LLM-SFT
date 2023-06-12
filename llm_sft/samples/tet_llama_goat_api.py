# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/6 21:52
# @author  : Mo
# @function: goat-orginal-code-github: https://github.com/liutiedong/goat


import random
import time
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TORCH"] = "1"
CPU_NUMS = "9"
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from peft import (get_peft_model, LoraConfig, LoraModel)
from transformers import GenerationConfig
import torch

from pydantic import BaseModel
from fastapi import FastAPI
import time
app = FastAPI()

from llm_sft.models.llama.model import LlamaForCausalLM, LlamaModel
from llm_sft.models.llama.tokenization_llama import LlamaTokenizer
# from transformers import LlamaForCausalLM, LlamaModel
# from transformers import LlamaTokenizer, LlamaConfig


LORA_DROPOUT = 0.05
LORA_ALPHA = 16
LORA_R = 8
SAVE_STEPS = 382
VAL_SET_SIZE = 0
MAX_LENGTH_Q = 256 - 2  # default=128 - 2
MAX_LENGTH_A = 256 - 2  # default=128 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 2
# PATH_TOKENIZER_PRETRAIN = "hf-internal-testing/llama-tokenizer"
# PATH_MODEL_PRETRAIN = "decapoda-research/llama-7b-hf"
# PATH_LORA_PRETRAIN = "tiedong/goat-lora-7b"
USE_CUDA = False if os.environ["CUDA_VISIBLE_DEVICES"]=="-1" else True
PATH_TOKENIZER_PRETRAIN = "hf-internal-testing/llama-tokenizer"
PATH_MODEL_PRETRAIN = "decapoda-research/llama-7b-hf"
PATH_LORA_PRETRAIN = "tiedong_goat/lora-7b"


def load_model_state(model, model_save_dir="./", model_name="adapter_model.bin", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode = True
        model = get_peft_model(model, peft_config)
        state_dict = torch.load(path_model, map_location=torch.device(device))
        # print(state_dict.keys())
        model.load_state_dict(state_dict, strict=False)
        # model.to(device)
        print("******model loaded success******")
        print("self.device: {}".format(device))
    except Exception as e:
        print(str(e))
        raise Exception("******load model error******")
    return model
def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.bin"):
    """  仅保存模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if config:
        config.save_pretrained(model_save_dir)
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    # torch.save(model.state_dict(), path_model)
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
                        if v.requires_grad == True}
    torch.save(grad_params_dict, path_model)
    print("******model_save_path is {}******".format(path_model))
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model
def print_named_parameters(model, use_print_data=True):
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
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"

    text_1 = f"{data_point.get('instruction', '')}\nAnswer: "
    text_2 = f"{data_point.get('output', '')}"

    x = tokenizer.encode(text_1.replace(" ", ""))
    y = tokenizer.encode(text_2.replace(" ", ""))
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
    return {"input_ids": x,
            "labels": y}


tokenizer = LlamaTokenizer.from_pretrained(PATH_TOKENIZER_PRETRAIN, add_eos_token=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"  # Allow batched inference
ID_PAD = tokenizer.convert_tokens_to_ids(["<pad>"])[0]
ID_UNK = tokenizer.convert_tokens_to_ids(["<unk>"])[0]
ID_BOS = tokenizer.convert_tokens_to_ids(["<s>"])[0]
ID_EOS = tokenizer.convert_tokens_to_ids(["</s>"])[0]
print(ID_PAD)
print(ID_UNK)
print(ID_BOS)
print(ID_EOS)

model = LlamaForCausalLM.from_pretrained(PATH_MODEL_PRETRAIN)
model = load_model_state(model=model, model_save_dir=PATH_LORA_PRETRAIN)
print("load peft ok")
model = prepare_model_for_half_training(model,
        use_gradient_checkpointing=True,
        output_embedding_layer_name="lm_head",
        layer_norm_names=["post_attention_layernorm",
                          "input_layernorm",
                          "norm",
                          ],
        )
if USE_CUDA:
    model = model.cuda()
else:
    model = model.bfloat16()
print_named_parameters(model, use_print_data=True)



def predict(data_dict, generation_dict):
    """  推理  """
    prompt_dict = generate_prompt(data_dict)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
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
            # max_new_tokens=512,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print(data_dict)
    print(input_ids)
    print(output)
    # output = output.split("答：")[-1]
    return output


class Item(BaseModel):
    instruction: str = "对下面数学题点评。"
    text: str = "1+1="
    max_new_tokens: int = 256
    temperature: float = 0.1
    do_sample: bool = False
    num_beams: int = 6
    top_p: float = 1.0
    top_k: int = 320


@app.post("/nlp/text_maths_goat")
def text_goat(request_data: Item):
    instruction = request_data.instruction
    text = request_data.text
    max_new_tokens = request_data.max_new_tokens
    temperature = request_data.temperature
    do_sample = request_data.do_sample
    num_beams = request_data.num_beams
    top_p = request_data.top_p
    top_k = request_data.top_k

    generation_dict = vars(request_data)
    generation_dict.pop("max_new_tokens")
    generation_dict.pop("instruction")
    generation_dict.pop("text")

    # request打印, 聊天内容包含敏感信息, 不打印
    if text:
        # logger.info("fastapi post:" + str(text).replace("\n", "")[:64])
        print(generation_dict)
        print("fastapi post:" + str(text).replace("\n", "")[:64])
    assert type(text) == str, {"code": 400, "data": {"details": "type of text must be string"}, "message": "fail"}
    try:  # 数据预处理, 模型预测
        # res = predict(instruction=instruction, text=text,
        #               max_new_tokens=max_new_tokens,
        #               generation_dict=generation_dict)
        generation_dict = {"max_new_tokens": max_new_tokens,
                           "temperature": temperature,
                           "do_sample": do_sample,
                           "num_beams": num_beams,
                           "top_p": top_p,
                           "top_k": top_k,
                           }
        ques_dict = {"instruction": instruction, "input": text, "output": ""}
        res = predict(ques_dict, generation_dict)
        res = {"res": res}
        code = 200
        message = "success"
    except Exception as e:
        torch.cuda.empty_cache()
        # logger.info(str(e))
        # print("fastapi post:" + str(text).replace("\n", "")[:64])
        res = {"res": {}}
        code = 400
        message = "fail"
    return {"code": code,
            "data": res,
            "message": message}


if __name__ == '__main__':
    ee = 0

    import uvicorn
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8835,
                workers=1)



"""
# nohup python tet_llama_goat_api.py > tc.tet_llama_goat_api.py.log 2>&1 &
# tail -n 1000  -f tc.tet_llama_goat_api.py.log
# |myz|


{
  "instruction": "1+1+1+1=",
  "text": "",
  "max_new_tokens": 256,
  "temperature": 0.1,
  "do_sample": true,
  "num_beams": 1,
  "top_p": 0.75,
  "top_k": 50
}

缺点: 只有整数-大数一元计算(单个计算符), 没有小数/分数/无理数等, 也没有多元运算(多个计算符)
"""

