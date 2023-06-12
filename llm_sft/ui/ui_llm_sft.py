# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/7 21:16
# @author  : Mo
# @function: ui-llm-sft
# @code    : most code from: https://huggingface.co/spaces/multimodalart/ChatGLM-6B/tree/main


from transformers import AutoModel, AutoTokenizer
import gradio as gr


# REPO_ID = "THUDM/chatglm-6b"
# tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
# model = AutoModel.from_pretrained(REPO_ID, trust_remote_code=True).half().cuda()
# model = model.eval()

from llm_sft.ft_bert.inference import BertSFT
PATH_MODEL_PRETRAIN = ""
REPO_ID = "hfl/chinese-roberta-wwm-ext"
PATH_MODEL_PRETRAIN = PATH_MODEL_PRETRAIN if PATH_MODEL_PRETRAIN else REPO_ID
model = BertSFT(PATH_MODEL_PRETRAIN=PATH_MODEL_PRETRAIN,
                   MODEL_SAVE_DIR="../ft_bert/model_sft",
                   USE_CUDA=True)


def predict(query, history=None):
    print(query)
    print(history)
    if history is None:
        history = []
    response, history = model.chat(query, history)
    return history, history


with gr.Blocks() as demo:
    gr.Markdown('''## ChatGLM-6B - demo''')
    state = gr.State([])
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=400)
    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder=
                             "Enter text and press enter").style(container=False)
        with gr.Column(scale=1):
            button = gr.Button("Generate")
    txt.submit(predict, [txt, state], [chatbot, state])
    button.click(predict, [txt, state], [chatbot, state])
demo.queue().launch()

