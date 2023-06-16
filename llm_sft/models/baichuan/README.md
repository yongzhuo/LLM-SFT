---
language:
- zh
- en
pipeline_tag: text-generation
inference: false
---
# baichuan-7B

<!-- Provide a quick summary of what the model is/does. -->

baichuan-7B是由百川智能开发的一个开源的大规模预训练模型。基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。在标准的中文和英文权威benchmark（C-EVAL/MMLU）上均取得同尺寸最好的效果。

如果希望使用baichuan-7B（如进行推理、Finetune等），我们推荐使用配套代码库[baichuan-7B](https://github.com/baichuan-inc/baichuan-7B)。

baichuan-7B is an open-source large-scale pre-trained model developed by Baichuan Intelligent Technology. Based on the Transformer architecture, it is a model with 7 billion parameters trained on approximately 1.2 trillion tokens. It supports both Chinese and English, with a context window length of 4096. It achieves the best performance of its size on standard Chinese and English authoritative benchmarks (C-EVAL/MMLU).

If you wish to use baichuan-7B (for inference, finetuning, etc.), we recommend using the accompanying code library [baichuan-7B](https://github.com/baichuan-inc/baichuan-7B).

## Why use baichuan-7B

- 在同尺寸模型中baichuan-7B达到了目前SOTA的水平，参考下面MMLU指标
- baichuan-7B使用自有的中英文双语语料进行训练，在中文上进行优化，在C-Eval达到SOTA水平
- 不同于LLaMA完全禁止商业使用，baichuan-7B使用更宽松的开源协议，允许用于商业目的

- Among models of the same size, baichuan-7B has achieved the current state-of-the-art (SOTA) level, as evidenced by the following MMLU metrics.
- baichuan-7B is trained on proprietary bilingual Chinese-English corpora, optimized for Chinese, and achieves SOTA performance on C-Eval.
- Unlike LLaMA, which completely prohibits commercial use, baichuan-7B employs a more lenient open-source license, allowing for commercial purposes.

## How to Get Started with the Model

如下是一个使用baichuan-7B进行1-shot推理的任务，根据作品给出作者名，正确输出为"夜雨寄北->李商隐"
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", device_map="auto", trust_remote_code=True)
inputs = tokenizer('登鹳雀楼->王之涣\n夜雨寄北->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

The following is a task of performing 1-shot inference using baichuan-7B, where the author's name is given based on the work, with the correct output being "One Hundred Years of Solitude->Gabriel Garcia Marquez"
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", device_map="auto", trust_remote_code=True)
inputs = tokenizer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64,repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
```

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Developed by:** 百川智能(Baichuan Intelligent Technology)
- **Email**: opensource@baichuan-inc.com
- **Language(s) (NLP):** Chinese/English
- **License:** [baichuan-7B License](https://huggingface.co/baichuan-inc/baichuan-7B/blob/main/baichuan-7B%20%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)

### Model Sources

<!-- Provide the basic links for the model. -->

整体模型基于标准的Transformer结构，我们采用了和LLaMA一样的模型设计
- **Position Embedding**：采用rotary-embedding，是现阶段被大多数模型采用的位置编码方案，具有很好的外推性。
- **Feedforward Layer**：采用SwiGLU，Feedforward变化为(8/3)倍的隐含层大小，即11008。
- **Layer Normalization**: 基于[RMSNorm](https://arxiv.org/abs/1910.07467)的Pre-Normalization。

具体参数和见下表
| Hyperparameter | Value |
|----------------|-------|
|n_parameters | 7000559616 |
|n_layers | 32 |
| n_heads | 32 |
| d_model | 4096 |
| vocab size | 64000 |
| sequence length | 4096 |

The overall model is based on the standard Transformer structure, and we have adopted the same model design as LLaMA:

- Position Embedding: We use rotary-embedding, which is the position encoding scheme adopted by most models at this stage, and it has excellent extrapolation capabilities.
- Feedforward Layer: We use SwiGLU. The feedforward changes to (8/3) times the size of the hidden layer, that is, 11008.
- Layer Normalization: Pre-Normalization based on [RMSNorm](https://arxiv.org/abs/1910.07467).

The specific parameters are as follows:
| Hyperparameter | Value |
|----------------|-------|
|n_parameters | 7000559616 |
|n_layers | 32 |
| n_heads | 32 |
| d_model | 4096 |
| vocab size | 64000 |
| sequence length | 4096 |

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Downstream Use 

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
我们同时开源出了和本模型配套的训练代码，允许进行高效的Finetune用于下游任务，具体参见[baichuan-7B](https://github.com/baichuan-inc/baichuan-7B)。

We have also open-sourced the training code that accompanies this model, allowing for efficient finetuning for downstream tasks. For more details, please refer to [baichuan-7B](https://github.com/baichuan-inc/baichuan-7B).

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
在没有充分评估风险和采取缓解措施的情况下投入生产使用；任何可能被视为不负责任或有害的使用案例。

Production use without adequate assessment of risks and mitigation; any use cases which may be considered irresponsible or harmful.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

baichuan-7B可能会产生事实上不正确的输出，不应依赖它产生事实上准确的信息。baichuan-7B是在各种公共数据集上进行训练的。尽管我们已经做出了巨大的努力来清洗预训练数据，但这个模型可能会生成淫秽、偏见或其他冒犯性的输出。

baichuan-7B can produce factually incorrect output, and should not be relied on to produce factually accurate information. baichuan-7B was trained on various public datasets. While great efforts have been taken to clean the pretraining data, it is possible that this model could generate lewd, biased or otherwise offensive outputs.

## Training Details

训练具体设置参见[baichuan-7B](https://github.com/baichuan-inc/baichuan-7B)。

For specific training settings, please refer to [baichuan-7B](https://github.com/baichuan-inc/baichuan-7B).

## Evaluation

### 中文评测
#### C-Eval
[CEval数据集](https://cevalbenchmark.com/index.html)是一个全面的中文基础模型评测数据集，涵盖了52个学科和四个难度的级别。我们使用该数据集的dev集作为few-shot的来源，在test集上进行了5-shot测试。


| Model 5-shot                | Average | Avg(Hard) | STEM | Social Sciences | Humanities | Others |
|-----------------------------|---------|-----------|------|-----------------|------------|--------|
| GPT-4                       | 68.7    | 54.9      | 67.1 | 77.6            | 64.5       | 67.8   |
| ChatGPT                     | 54.4    | 41.4      | 52.9 | 61.8            | 50.9       | 53.6   |
| Claude-v1.3                 | 54.2    | 39.0      | 51.9 | 61.7            | 52.1       | 53.7   |
| Claude-instant-v1.0         | 45.9    | 35.5      | 43.1 | 53.8            | 44.2       | 45.4   |
| moss-moon-003-base (16B)    | 27.4    | 24.5      | 27.0 | 29.1            | 27.2       | 26.9   |
| Ziya-LLaMA-13B-pretrain     | 30.2    | 22.7      | 27.7 | 34.4            | 32.0       | 28.9   |
| LLaMA-7B-hf                 | 27.1    | 25.9      | 27.1 | 26.8            | 27.9       | 26.3   |
| ChatGLM-6B                  | 34.5    | 23.1      | 30.4 | 39.6            | 37.4       | 34.5   |
| Falcon-7B                   | 25.8    | 24.3      | 25.8 | 26.0            | 25.8       | 25.6   |
| Open-LLaMA-v2-pretrain (7B) | 24.0    | 22.5      | 23.1 | 25.3            | 25.2       | 23.2   |
| TigerBot-7B-base            | 25.7    | 27.0      | 27.3 | 24.7            | 23.4       | 26.1   |
| Aquila-7B<sup>*</sup>       | 25.5    | 25.2      | 25.6 | 24.6            | 25.2       | 26.6   |
| BLOOM-7B                    | 22.8    | 20.2      | 21.8 | 23.3            | 23.9       | 23.3   |
| BLOOMZ-7B                   | 35.7    | 25.8      | 31.3 | 43.5            | 36.6       | 35.6   |
| **baichuan-7B**             | 42.8    | 31.5      | 38.2 | 52.0            | 46.2       | 39.3   |


#### Gaokao
[Gaokao](https://github.com/ExpressAI/AI-Gaokao) 是一个以中国高考题作为评测大语言模型能力的数据集，用以评估模型的语言能力和逻辑推理能力。
我们只保留了其中的单项选择题，并对所有模型进行统一5-shot测试。

以下是测试的结果。

| Model           | Average |
|-------------------------|-----------------|
| Open-LLaMA-v2-pretrain  | 21.41           |
| Ziya-LLaMA-13B-pretrain | 23.17           |
| Falcon-7B               | 23.98           |
| TigerBot-7B-base        | 25.94           |
| LLaMA-7B                | 27.81           |
| ChatGLM-6B              | 21.41           |
| BLOOM-7B                | 26.96           |
| BLOOMZ-7B               | 28.72           |
| Aquila-7B<sup>*</sup>               | 24.39           |
| **baichuan-7B**        | **36.24**           |


#### AGIEval
[AGIEval](https://github.com/microsoft/AGIEval) 旨在评估模型的认知和解决问题相关的任务中的一般能力。
我们只保留了其中的四选一单项选择题，随机划分后对所有模型进行了统一5-shot测试。

| Model           | Average |
|-------------------------|-----------------|
| Open-LLaMA-v2-pretrain  | 23.49           |
| Ziya-LLaMA-13B-pretrain | 27.64           |
| Falcon-7B               | 27.18           |
| TigerBot-7B-base        | 25.19           |
| LLaMA-7B                | 28.17           |
| ChatGLM-6B              | 23.49           |
| BLOOM-7B                | 26.55           |
| BLOOMZ-7B               | 30.27           |
| Aquila-7B<sup>*</sup>               | 25.58           |
| **baichuan-7B**        | **34.44**           |

<sup>*</sup>其中Aquila模型来源于[智源官方网站](https://model.baai.ac.cn/model-detail/100098)，仅做参考

### English Leaderboard
In addition to Chinese, we also tested the model's performance in English. 

#### MMLU

[MMLU](https://arxiv.org/abs/2009.03300) is an English evaluation dataset that includes 57 multiple-choice tasks, covering elementary mathematics, American history, computer science, law, etc. The difficulty ranges from high school level to expert level, making it a mainstream LLM evaluation dataset.

We adopted the [open-source]((https://github.com/hendrycks/test)) evaluation scheme, and the final 5-shot results are as follows:

| Model                                  | Humanities | Social Sciences | STEM | Other | Average |
|----------------------------------------|-----------:|:---------------:|:----:|:-----:|:-------:|
| LLaMA-7B<sup>2</sup>                   |       34.0 |      38.3       | 30.5 | 38.1  |  35.1   |
| Falcon-7B<sup>1</sup>                  |          - |        -        |  -   |   -   |  35.0   |
| mpt-7B<sup>1</sup>                     |          - |        -        |  -   |   -   |  35.6   |
| ChatGLM-6B<sup>0</sup>                 |       35.4 |      41.0       | 31.3 | 40.5  |  36.9   |
| BLOOM 7B<sup>0</sup>                  |       25.0 |      24.4       | 26.5 | 26.4  |  25.5   |
| BLOOMZ 7B<sup>0</sup>                 |       31.3 |      42.1       | 34.4 | 39.0  |  36.1   |
| moss-moon-003-base (16B)<sup>0</sup>   |       24.2 |      22.8       | 22.4 | 24.4  |  23.6   |
| moss-moon-003-sft (16B)<sup>0</sup>    |       30.5 |      33.8       | 29.3 | 34.4  |  31.9   |
| **baichuan-7B<sup>0</sup>**            |       38.4 |      48.9       | 35.6 | 48.1  |  42.3   |

The superscript in the Model column indicates the source of the results.
```
0:reimplemented
1:https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
2:https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
```
