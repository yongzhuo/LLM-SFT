# LLM-SFT
中文大模型微调(LLM-SFT), 支持模型(ChatGLM, LlaMA, Bloom, Baichuan-7B), 支持(LoRA, QLoRA, DeepSpeed, UI, TensorboardX), 支持(微调, 推理, 测评, 接口)等.

## 踩坑
```python
LoRA: ChatGLM已经微调比较好了, 垂直领域数据继续微调甚至会带来性能下降, 建议至多不超过200w-epoch(R=8的情况);
QLoRA: 不要使用.cuda(), GPU至少为英伟达图灵架构往上【备注】当前(2023.06)QLoRA只是节约显存, 并不能加速训练;
```

## LoRA权重
```shell
Bloomz-7B-GPT4ForALL: https://huggingface.co/Macropodus/MWP-Instruct
ChatGLM-6B-GPT4ForALL: https://huggingface.co/Macropodus/MWP-Instruct
LlaMA-7B-GPT4ForALL: https://huggingface.co/Macropodus/MWP-Instruct
ChatGLM-6B-MWP: https://huggingface.co/Macropodus/MWP-Instruct
```

## 微调数据
1. 原始数据来自[https://github.com/LYH-YF/MWPToolkit](https://github.com/LYH-YF/MWPToolkit)

处理后的微调数据(多步计算+一/二元解方程)-MWP: [https://huggingface.co/datasets/Macropodus/MWP-Instruct](https://huggingface.co/datasets/Macropodus/MWP-Instruct)


2. 大数加减乘除来自: [https://github.com/liutiedong/goat.git ](https://github.com/liutiedong/goat.git )

## 微调样例
```shell
地址: llm_sft/ft_chatglm

配置: llm_sft/ft_chatglm/config.py
训练: python train.py
推理: python predict.py
验证: python evaluation.py
接口: python post_api.py

```

## 环境配置
```shell
1.详见LLM-SFT/requirements.txt
transformers>=4.26.1
torch>=1.10.1
peft>=0.2.0

2.注意QLoRA需要的版本更高些, 详见LLM-SFT/llm_sft/ft_qlora/requirements.txt
transformers>=4.30.0.dev0
accelerate>=0.20.0.dev0
bitsandbytes>=0.39.0
peft>=0.4.0.dev0
torch>=1.13.1
```

## 数据集-中文
 - [https://huggingface.co/datasets/JosephusCheung/GuanacoDataset](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)
 - [https://huggingface.co/datasets/shareAI/shareGPT_cn](https://huggingface.co/datasets/shareAI/shareGPT_cn)
 - [https://huggingface.co/datasets/Mutonix/RefGPT-Fact](https://huggingface.co/datasets/Mutonix/RefGPT-Fact)
 - [https://huggingface.co/datasets/BAAI/COIG](https://huggingface.co/datasets/BAAI/COIG)
 - [https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
 - [https://github.com/carbonz0/alpaca-chinese-dataset](https://github.com/carbonz0/alpaca-chinese-dataset)
 - [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
 - [https://github.com/PhoebusSi/Alpaca-CoT](https://github.com/PhoebusSi/Alpaca-CoT)
 - [https://github.com/Hello-SimpleAI/chatgpt-comparison-detection](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)
 - [https://github.com/yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)
 - [https://github.com/XueFuzhao/InstructionWild](https://github.com/XueFuzhao/InstructionWild)
 - [https://github.com/OpenLMLab/MOSS](https://github.com/OpenLMLab/MOSS)
 - [https://github.com/thu-coai/Safety-Prompts](https://github.com/thu-coai/Safety-Prompts)
 - [https://github.com/LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
 - [https://github.com/TigerResearch/TigerBot](https://github.com/TigerResearch/TigerBot)

## 参考/感谢
 - [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
 - [https://github.com/THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
 - [https://github.com/THUDM/GLM](https://github.com/THUDM/GLM)
 - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
 - [https://github.com/LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
 - [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
 - [https://github.com/mymusise/ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)
 - [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
 - [https://github.com/bojone/bert4keras](https://github.com/bojone/bert4keras)
 - [trl](https://github.com/lvwerra/trl)
 - [https://github.com/LYH-YF/MWPToolkit](https://github.com/LYH-YF/MWPToolkit)
 - [math23k](https://aclanthology.org/D17-1088)
 - [https://github.com/ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
 - [https://github.com/bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)
 - [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)
 - [https://huggingface.co/spaces/multimodalart/ChatGLM-6B/tree/main](https://huggingface.co/spaces/multimodalart/ChatGLM-6B/tree/main)
 - [https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat/tree/main](https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat/tree/main)
 - [https://github.com/artidoro/qlora](https://github.com/artidoro/qlora)
 - [https://github.com/baichuan-inc/baichuan-7B](https://github.com/baichuan-inc/baichuan-7B)

# Reference
For citing this work, you can refer to the present GitHub project. For example, with BibTeX:
```
@misc{Keras-TextClassification,
    howpublished = {\url{https://github.com/yongzhuo/LLM-SFT}},
    title = {LLM-SFT},
    author = {Yongzhuo Mo},
    publisher = {GitHub},
    year = {2023}
}
```

## 免责申明
本项目相关资源仅供学术研究之用，严禁用于商业用途。 使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。
 - 大模型权重的详细协议见[THUDM/chatglm-6b](https://github.com/THUDM/ChatGLM-6B),  [bigscience/bloomz-7b1-mt](https://github.com/bigscience-workshop/petals),  [decapoda-research/llama-7b-hf](https://github.com/facebookresearch/llama) 
 - 指令微调数据协议见[GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM),  [LYH-YF/MWPToolkit](https://github.com/LYH-YF/MWPToolkit),  [yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)

