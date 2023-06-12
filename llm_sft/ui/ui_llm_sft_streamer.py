# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/7 21:53
# @author  : Mo
# @function: LLM-SFT-streamer
# @code    : most code from: https://huggingface.co/spaces/stabilityai/stablelm-tuned-alpha-chat/tree/main


from threading import Thread
import time
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(path_root)
sys.path.append(path_root)

from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.nn import functional as F
from queue import Queue
import gradio as gr
import numpy as np
import torch


print(f"Starting to load the model to memory")
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
tokenizer = model.tokenizer
model = model.model
print(f"Sucessfully loaded the model to the memory")
start_message = """<|SYSTEM|># LLM-SFT-Assistant."""


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [13005]
        # stop_ids = [105]
        # stop_ids = [50278, 50279, 50277, 1, 0]
        # stop_ids = [105, 102]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def chat(curr_system_message, history):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = curr_system_message + \
        "".join(["".join(["<|USER|>"+item[0], "<|ASSISTANT|>"+item[1]]) for item in history])

    # Tokenize the messages string
    model_inputs = tokenizer([messages], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10.,
                                    skip_special_tokens=True,
                                    skip_prompt=True,)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        # max_length=510,
        temperature=0.95,
        do_sample=True,
        top_p=0.7,
        top_k=50,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # print(history)
    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        # print(new_text)
        partial_text += new_text
        history[-1][1] = partial_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        yield history
    return partial_text


with gr.Blocks() as demo:
    # history = gr.State([])
    gr.Markdown("## LLM-SFT-Chat")
    gr.HTML('''<center>Duplicate the Space to skip the queue and run in a private space</center>''')
    chatbot = gr.Chatbot().style(height=400)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(label="Chat Message Box", placeholder="Chat Message Box",
                             show_label=False).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    system_msg = gr.Textbox(start_message, label="System Message",
                            interactive=False, visible=False)

    submit_event = msg.submit(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        .then(fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
    submit_click_event = submit.click(fn=user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)\
        .then(fn=chat, inputs=[system_msg, chatbot], outputs=[chatbot], queue=True)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_event, submit_click_event], queue=False)
    clear.click(lambda: None, None, [chatbot], queue=False)

demo.queue(max_size=32, concurrency_count=2)
demo.launch()

