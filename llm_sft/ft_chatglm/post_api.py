# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/25 21:56
# @author  : Mo
# @function: fastapi-post接口


import traceback
import logging
import random
import time
import json
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)
from llm_sft.ft_chatglm.config import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model
from transformers import GenerationConfig
from pydantic import BaseModel
from rouge import Rouge  # pip install rouge
from tqdm import tqdm
import torch

from pydantic import BaseModel
from fastapi import FastAPI
import time

# from transformers import ChatGLMForConditionalGeneration, ChatGLMConfig
# from transformers import ChatGLMTokenizer
from llm_sft.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration, ChatGLMConfig
from llm_sft.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from llm_sft.ft_chatglm.config import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from llm_sft.ft_chatglm.config import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from llm_sft.ft_chatglm.config import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from llm_sft.ft_chatglm.config import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from llm_sft.ft_chatglm.config import LORA_DROPOUT, LORA_ALPHA, LORA_R


app = FastAPI()  # 日志文件名,为启动时的日期, 全局日志格式
logger_level = logging.INFO
logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] "
                           "- %(levelname)s: %(message)s",
                    level=logger_level)
logger = logging.getLogger("ft-chatglm")
console = logging.StreamHandler()
console.setLevel(logger_level)
logger.addHandler(console)


# device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
device_map = "auto"
USE_CUDA = True
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
def load_model_state(model, model_save_dir="./", model_name="adapter_model.bin", device="cpu"):
    """  仅加载模型参数(推荐使用)  """
    try:
        path_model = os.path.join(model_save_dir, model_name)
        peft_config = LoraConfig.from_pretrained(model_save_dir)
        peft_config.inference_mode = True
        model = get_peft_model(model, peft_config)
        state_dict = torch.load(path_model, map_location=torch.device(device))
        print(state_dict.keys())
        model.load_state_dict(state_dict, strict=False)
        # model.to(device)
        print("******model loaded success******")
        print("self.device: {}".format(device))
    except Exception as e:
        print(str(e))
        raise Exception("******load model error******")
    return model
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
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
def generate_prompt(data_point):
    """  构建prompt   """
    if data_point["input"]:
        text_1, text_2 = f"""下面是一条指令。请根据问题，并编写一个准确的回答，以适当地完成指令。
        \n###指令：\n{data_point["instruction"]}
        \n###问题：\n{data_point["input"]}
        \n###回答：\n""", f"""{data_point["output"]}"""
    else:
        text_1, text_2 = f"""下面是一条指令。请编写一个准确的回答，以适当地完成指令。
        \n###指令：\n{data_point["instruction"]}
        \n###回答：\n""", f"""{data_point["output"]}"""

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
def data_collator(batch):
    # there's probably a way to do this with the tokenizer settings
    def get_position_ids(seq, bos_token_id, gmask=True, position_encoding_2d=True):
        """  code from model_chatglm.py  """
        # context_length = seq.index(bos_token_id) + 1
        context_length = len(seq)
        position_ids = torch.arange(context_length, dtype=torch.long)
        if position_encoding_2d:
            seq_length = seq.index(bos_token_id)
            if not gmask:
                mask_position = seq_length - 1
                position_ids[seq_length:] = mask_position
            block_position_ids = torch.cat((
                torch.zeros(seq_length, dtype=torch.long),
                torch.arange(context_length - seq_length, dtype=torch.long) + 1
            ))
            position_ids = torch.stack((position_ids, block_position_ids), dim=0)
        else:
            if not gmask:
                seq_length = seq.index(bos_token_id)
                mask_position = seq_length - 1
                position_ids[context_length - 1:] = mask_position
        # position_ids = position_ids.unsqueeze(0)
        return position_ids

    def get_masks(seq, bos_token_id):
        """  code from model_chatglm.py  """
        context_length = seq.index(bos_token_id)
        attention_mask = torch.ones((1, len(seq), len(seq)))
        attention_mask.tril_()
        attention_mask[..., :context_length] = 1
        # attention_mask.unsqueeze_(1)
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels")) + 1
                     for i in range(len(batch))]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_position_ids = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        ## 382, 383
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y) 
        labels = [-100] * len(x) + y + [-100] * len_padding
        input_ids = x + y + [ID_PAD] * (len_padding)
        tensor_position_ids = get_position_ids(input_ids, ID_BOS, gmask=True,
                                               position_encoding_2d=True)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        tensor_attention_mask = get_masks(input_ids, ID_BOS)
        batch_attention_mask.append(tensor_attention_mask)
        batch_position_ids.append(tensor_position_ids)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    # print(batch_attention_mask)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_position_ids = torch.stack(batch_position_ids)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = {"attention_mask": batch_attention_mask,
                  "position_ids": batch_position_ids,
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    return input_dict


model = ChatGLMForConditionalGeneration.from_pretrained(PATH_MODEL_PRETRAIN)
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()
# model.is_parallelizable = False
# model.config.use_cache = False
# model.model_parallel = False
print_named_parameters(model, True)
tokenizer = ChatGLMTokenizer.from_pretrained(PATH_MODEL_PRETRAIN, add_eos_token=True)
ID_gMASK = 130001
ID_BOS = 130004
ID_EOS = 130005
ID_MASK = 130000
ID_PAD = 3
model = load_model_state(model=model, model_save_dir=MODEL_SAVE_DIR)
# model = prepare_model_for_half_training(model,
#         use_gradient_checkpointing=True,
#         output_embedding_layer_name="lm_head",
#         layer_norm_names=["post_attention_layernorm",
#                           "input_layernorm",
#                           "final_layernorm"
#                           ],
#         )
if USE_CUDA:
    model = model.half().cuda()
else:
    model = model.bfloat16()
print_named_parameters(model, True)


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
def predict(data_point, generation_config):
    """  推理  """
    prompt_dict = generate_prompt(data_point)
    # inputs = tokenizer([text_1], return_tensors="pt", padding=True)
    input_ids = prompt_dict.get("input_ids")
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    if USE_CUDA:
        input_ids = input_ids.cuda()
    generation_config = GenerationConfig(**generation_config)
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
    print(input_ids)
    print(s)
    print(output)
    return output
class Item(BaseModel):
    instruction: str = "完成下面的问答"
    text: str = "1+1="
    penalty_alpha: float = 1.0
    max_new_tokens: int = 512
    temperature: float = 0.95  # 0.95  # 0.35  # 0.95
    do_sample: bool = True
    num_beams: int = 1
    top_p: float = 0.7  # 0.75
    top_k: int = 50


@app.post("/nlg/text_generate")
def text_generate(request_data: Item):
    instruction = request_data.instruction
    text = request_data.text
    penalty_alpha = request_data.penalty_alpha
    max_new_tokens = request_data.max_new_tokens
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
    generation_config = {"temperature": temperature,
                         "top_p": top_p,
                         "top_k": top_k,
                         "num_beams": num_beams,
                         "do_sample": do_sample,
                         "penalty_alpha": penalty_alpha,
                         "max_new_tokens": max_new_tokens,
                         }
    try:  # 数据预处理, 模型预测
        response = predict(data_point, generation_config)
    except Exception as e:
        logger.info(traceback.print_exc())
        response = "[EOP]"
    return response


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,
                host="0.0.0.0",
                port=8032,
                workers=1)


"""
# nohup python post_api.py > tc.post_api.py.log 2>&1 &
# tail -n 1000  -f tc.post_api.py.log
# |myz|

可以在浏览器生成界面直接访问: http://localhost:8032/docs
"""