import gradio as gr
import gradio_image_prompter as gr_ext

import sys
import os
from text_prompt import Inference as TextInfer
from visual_prompt import Inference as VisualInfer
import threading
from easydict import EasyDict as edict
import numpy as np
import time
import tempfile


thread_local = threading.local()
kv = {}

args = edict()
args.config_file = "configs/text_model_cfgs.yaml"

def Py_Text(img, text):
    if img is None or text is None:
        return None

    def initialize_object():
        if not hasattr(thread_local, 'object'):
            thread_local.object = TextInfer(args, 'cuda:0')
        return thread_local.object

    obj = initialize_object()

    o1, o2 = obj.predict(img, text)
    return o1, o2


def get_examples():
    examples = [
        ["resources/fruit.jpg", "grape.hami melon.mango.cherry.orange.peach.pitaya.kiwi.watermelon"],
        ["resources/ships.jpg", "ships"],
    ]

    return examples

def get_points_from_prompter(click_img):
    points = click_img["points"]
    points = np.array(points).reshape((-1, 2, 3))

    points = points.reshape((-1, 3))
    lt = points[np.where(points[:, 2] == 2)[0]][None, :, :2]
    rb = points[np.where(points[:, 2] == 3)[0]][None, :, :2]
    points = [lt, rb]
    points = np.concatenate(points, axis=-1)
    points = points.reshape(-1).tolist()
    return points


def on_feats_btn(click_img0, click_img1):
    args_v = edict()
    args_v.config_file = "configs/visual_model_cfgs.yaml"

    def initialize_object():
        if not hasattr(thread_local, 'vobject'):
            thread_local.vobject = VisualInfer(args_v, 'cuda:0')
            thread_local.id = threading.current_thread().ident
            kv[thread_local.id] = thread_local.vobject
        return thread_local.vobject

    obj = initialize_object()

    progress = gr.Progress()

    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)

    if click_img0 is not None:
        points0 = get_points_from_prompter(click_img0)
        img0 = click_img0['image']
    else:
        points0 = None
        img0 = None
    if click_img1 is not None:
        points1 = get_points_from_prompter(click_img1)
        img1 = click_img1['image']
    else:
        points1 = None
        img1 = None

    v_feats = obj.get_visual_prompts([img0, img1], [points0, points1], progress)
    v_feats = v_feats.cpu().detach().numpy()

    os.makedirs('TMP', exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, dir="TMP") as temp_file:
        temp_file_name = temp_file.name + '.npy'
        np.save(temp_file_name, v_feats)

    return "Completed", temp_file_name

def on_submit_btn(target_img, v_feats_path):
    ident = threading.current_thread().ident

    if ident not in kv:
        args_v = edict()
        args_v.config_file = "configs/visual_model_cfgs.yaml"
        thread_local.vobject = VisualInfer(args_v, 'cuda:0')
        thread_local.id = ident
        kv[ident] = thread_local.vobject

    obj = kv[ident]
    v_feats = np.load(v_feats_path)
    out = obj.predict_img(target_img, v_feats)

    return out

with gr.Blocks() as demo:
    with gr.Tab('TextPrompt'):
        input_image = gr.Image(label='输入图像')
        text_prompt = gr.Textbox(lines=1, placeholder="person.cat.dog")

        det_output = gr.Image(label='检测结果')
        seg_output = gr.Image(label='分割&检测结果')
        py_text = gr.Interface(
            fn=Py_Text, inputs=[input_image, text_prompt],
            outputs=[det_output, seg_output],
        )

        gr.Examples(
            examples=get_examples(),
            fn=Py_Text,
            inputs=[input_image, text_prompt],
            outputs=[det_output, seg_output],
            cache_examples=True
        )
    with gr.Tab('VisualPrompt'):
        with gr.Row():
            with gr.Column():
                click_img0 = gr_ext.ImagePrompter(label="提示图0", show_label=True)
                click_img1 = gr_ext.ImagePrompter(label="提示图1", show_label=True)
                run_feats_btn = gr.Button("提取提示图的特征")
                status_bar = gr.Text(label="进度条")
                md5_bar = gr.Text(label='md5', visible=False)
            with gr.Column():
                target_img = gr.Image(label='目标图')
                output_img = gr.Image(label='输出')
                run_submit_btn = gr.Button("运行检测")

            run_feats_btn.click(on_feats_btn, [click_img0, click_img1], [status_bar, md5_bar])
            run_submit_btn.click(on_submit_btn, [target_img, md5_bar], [output_img])

demo.launch(share=False, server_name='0.0.0.0', server_port=7862)

