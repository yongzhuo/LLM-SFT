# ft-qlora
QLoRA, Int4微调, 使用bitsandbytes/peft/transformers

## 踩坑
 - [issues/41](https://github.com/artidoro/qlora/issues/41)    注意千万不要使用.cuda(), 否则会报错RuntimeError: mat1 and mat2 shapes cannot be multiplied 
 - GPU >= NVIDIA Turing. for example, int8/int4 do not support V100.
 - Inference is slow, maybe you can transform to fp16 with new lora and original weights.

## 环境配置
```shell
transformers==4.30.0.dev0
accelerate==0.20.0.dev0
bitsandbytes==0.39.0
peft==0.4.0.dev0
datasets==2.9.0
torch==1.13.1


```

## 微调
```shell
qlora
配置: llm_sft/ft_qlora/config.py
训练: python train.py
推理: python predict.py
验证: python evaluation.py
接口: python post_api.py

```


## 数据集-中文
 - [https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
 - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
 - [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
 - [https://github.com/carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset)


## 参考/感谢
 - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
 - [https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
 - [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)

