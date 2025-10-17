
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import argparse
import numpy as np
import torchvision
from PIL import Image, ImageDraw, ImageFont

import sys
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(this_dir, '..'))

import WeVisionOne
from WeVisionOne.utils import box_ops

from detectron2.config import get_cfg
from detectron2.modeling import build_backbone, BACKBONE_REGISTRY
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.utils.colormap import random_color, _COLORS

from common import add_config


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(cfg.MODEL.SEM_SEG_HEAD.NAME)(cfg, self.backbone.output_shape())
        self.text_backbone = BACKBONE_REGISTRY.get("CLIPLangEncoder")(cfg, None)

        self.proj_text_256 = torch.nn.Linear(768, 256)

    def forward(self, x, texts):
        backboneOutput = self.backbone(x)
        txt_embeds = self.text_backbone(texts)
        txt_embeds = self.proj_text_256(txt_embeds)

        outputs = self.sem_seg_head(backboneOutput, txt_feats=txt_embeds)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(x.shape[-2], x.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        # bs 1
        for mask_cls_result, mask_pred_result, mask_box_result in zip(mask_cls_results, mask_pred_results, mask_box_results):
            boxes = box_ops.box_cxcywh_to_xyxy(mask_box_result)
            img_w = x.shape[-1]
            img_h = x.shape[-2]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
            scale_fct = scale_fct.to(mask_box_result)
            mask_box_result = boxes * scale_fct

            scores = mask_cls_result.sigmoid()
            num_classes = scores.shape[1]
            labels = torch.arange(num_classes).unsqueeze(0).repeat(900, 1).flatten(0, 1).to(scores.device)
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(100, sorted=False)

            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // num_classes
            mask_pred = mask_pred_result[topk_indices]
            mask_box_result = mask_box_result[topk_indices]

            pred_masks = (mask_pred > 0).float()
            pred_boxes = mask_box_result
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)
            scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image

        return scores, pred_boxes, pred_masks, pred_classes


class Inference:

    def __init__(self, args, device):
        cfg = get_cfg()
        add_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.freeze()

        ckpt = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device('cpu'))
        self.model = Model(cfg)
        out = self.model.load_state_dict(ckpt, strict=False)
        print(out)

        self.model.eval()
        self.model.to(device)
        self.device = device

    def predict(self, img_rgb, text_prompts, scoreThres=0.3, iouThres=0.55):
        img = img_rgb
        text_prompts = text_prompts.split('.')
        text_prompts.append('none')
        texts = [text_prompts]

        img_height, img_width = img.shape[:2]
        draw_img = img.copy()

        max_size = max(img_height, img_width)
        M = np.array([
            [1024.0 / max_size, 0.0, 0.0],
            [0.0, 1024.0 / max_size, 0.0]]
        ).astype(np.float32)
        invM = cv2.invertAffineTransform(M)

        resized_img = cv2.warpAffine(img, M, (1024, 1024), flags=cv2.INTER_LINEAR)

        resized_img = np.ascontiguousarray(resized_img)
        resized_img = torch.from_numpy(resized_img).permute(2, 0, 1).float()

        resized_img[0, :, :] = (resized_img[0, :, :] - 123.6750) / 58.3950
        resized_img[1, :, :] = (resized_img[1, :, :] - 116.2800) / 57.1200
        resized_img[2, :, :] = (resized_img[2, :, :] - 103.5300) / 57.3750

        resized_img = resized_img.to(self.device)
        resized_img = resized_img.unsqueeze(0)

        with torch.no_grad():
            scores, pred_boxes, pred_masks, pred_classes = self.model(resized_img, texts)

        keeps = scores > scoreThres
        scores = scores[keeps]
        pred_boxes = pred_boxes[keeps]
        pred_masks = pred_masks[keeps]
        pred_classes = pred_classes[keeps]

        nms_keeps = torchvision.ops.nms(pred_boxes, scores, iou_threshold=iouThres)

        scores = scores[nms_keeps].detach().cpu().numpy()
        pred_boxes = pred_boxes[nms_keeps].detach().cpu().numpy()
        pred_masks = pred_masks[nms_keeps].detach().cpu().numpy()
        pred_classes = pred_classes[nms_keeps].detach().cpu().numpy()

        image_pil = Image.fromarray(np.uint8(draw_img))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()

        ret = [_COLORS[i] * 255 for i in range(len(_COLORS))]
        num_remain = 1000 - len(ret)
        thing_colors = ret + [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(num_remain)]

        info = []
        for index, (score, pred_box, pred_mask, pred_class) in enumerate(zip(scores, pred_boxes, pred_masks, pred_classes)):
            pred_box = np.clip(pred_box, 0, 1024)
            xmin = pred_box[0] * invM[0, 0] + pred_box[1] * invM[0, 1] + invM[0, 2]
            ymin = pred_box[0] * invM[1, 0] + pred_box[1] * invM[1, 1] + invM[1, 2]
            xmax = pred_box[2] * invM[0, 0] + pred_box[3] * invM[0, 1] + invM[0, 2]
            ymax = pred_box[2] * invM[1, 0] + pred_box[3] * invM[1, 1] + invM[1, 2]

            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            color = thing_colors[pred_class]
            pred_cls_text = texts[0][pred_class] + '({:.2f})'.format(score)

            draw.rectangle([xmin, ymin, xmax, ymax], outline=tuple(color), width=3)

            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((xmin, ymin), str(pred_cls_text), font)
            else:
                w, h = draw.textsize(str(pred_cls_text), font)
                bbox = (xmin, ymin, w + xmin, ymin + h)
            draw.rectangle(bbox, fill=tuple(color))
            draw.text((xmin, ymin), str(pred_cls_text), fill="white")

            info.append([pred_cls_text])

        draw_img = np.array(image_pil)
        det_img = draw_img.copy()

        for index, (score, pred_box, pred_mask, pred_class) in enumerate(zip(scores, pred_boxes, pred_masks, pred_classes)):
            color = thing_colors[pred_class]

            bg = np.ones_like(draw_img)
            bg[:, :, 0] = color[2]
            bg[:, :, 1] = color[1]
            bg[:, :, 2] = color[0]

            pred_mask = cv2.warpAffine(pred_mask, invM, (img_width, img_height), flags=cv2.INTER_LINEAR)
            matts = 0.5 * pred_mask[:, :, None]
            draw_img = draw_img * (1.0 - matts) + bg * matts

        mask_img = draw_img.astype(np.uint8)

        return det_img, mask_img


    def predict_with_args(self, args):
        img = cv2.imread(args.img_path)
        text_prompts = args.text_prompts.split('.')
        text_prompts.append('none')
        texts = [text_prompts]

        img_height, img_width = img.shape[:2]
        draw_img = img.copy()

        max_size = max(img_height, img_width)
        M = np.array([
            [1024.0 / max_size, 0.0, 0.0],
            [0.0, 1024.0 / max_size, 0.0]]
        ).astype(np.float32)
        invM = cv2.invertAffineTransform(M)

        resized_img = cv2.warpAffine(img, M, (1024, 1024), flags=cv2.INTER_LINEAR)

        resized_img = resized_img[:, :, ::-1]
        resized_img = np.ascontiguousarray(resized_img)
        resized_img = torch.from_numpy(resized_img).permute(2, 0, 1).float()

        resized_img[0, :, :] = (resized_img[0, :, :] - 123.6750) / 58.3950
        resized_img[1, :, :] = (resized_img[1, :, :] - 116.2800) / 57.1200
        resized_img[2, :, :] = (resized_img[2, :, :] - 103.5300) / 57.3750

        resized_img = resized_img.to(self.device)
        resized_img = resized_img.unsqueeze(0)

        with torch.no_grad():
            scores, pred_boxes, pred_masks, pred_classes = self.model(resized_img, texts)

        scoreThres = args.scoreThres
        keeps = scores > scoreThres
        scores = scores[keeps]
        pred_boxes = pred_boxes[keeps]
        pred_masks = pred_masks[keeps]
        pred_classes = pred_classes[keeps]

        nms_keeps = torchvision.ops.nms(pred_boxes, scores, iou_threshold=args.iouThres)

        scores = scores[nms_keeps].detach().cpu().numpy()
        pred_boxes = pred_boxes[nms_keeps].detach().cpu().numpy()
        pred_masks = pred_masks[nms_keeps].detach().cpu().numpy()
        pred_classes = pred_classes[nms_keeps].detach().cpu().numpy()

        image_pil = Image.fromarray(np.uint8(draw_img[:, :, ::-1]))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.load_default()

        ret = [_COLORS[i] * 255 for i in range(len(_COLORS))]
        num_remain = 100 - len(ret)
        thing_colors = ret + [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(num_remain)]

        info = []
        for index, (score, pred_box, pred_mask, pred_class) in enumerate(zip(scores, pred_boxes, pred_masks, pred_classes)):
            pred_box = np.clip(pred_box, 0, 1024)
            xmin = pred_box[0] * invM[0, 0] + pred_box[1] * invM[0, 1] + invM[0, 2]
            ymin = pred_box[0] * invM[1, 0] + pred_box[1] * invM[1, 1] + invM[1, 2]
            xmax = pred_box[2] * invM[0, 0] + pred_box[3] * invM[0, 1] + invM[0, 2]
            ymax = pred_box[2] * invM[1, 0] + pred_box[3] * invM[1, 1] + invM[1, 2]

            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            color = thing_colors[pred_class]
            pred_cls_text = texts[0][pred_class] + '({:.2f})'.format(score)

            draw.rectangle([xmin, ymin, xmax, ymax], outline=tuple(color), width=3)

            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((xmin, ymin), str(pred_cls_text), font)
            else:
                w, h = draw.textsize(str(pred_cls_text), font)
                bbox = (xmin, ymin, w + xmin, ymin + h)
            draw.rectangle(bbox, fill=tuple(color))
            draw.text((xmin, ymin), str(pred_cls_text), fill="white")

            info.append([pred_cls_text])

        draw_img = np.array(image_pil)[:, :, ::-1]
        out_path = os.path.join(args.output_dir, "pred.png")
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(out_path):
            cv2.imwrite(out_path, draw_img)

        for index, (score, pred_box, pred_mask, pred_class) in enumerate(zip(scores, pred_boxes, pred_masks, pred_classes)):
            color = thing_colors[pred_class]

            bg = np.ones_like(draw_img)
            bg[:, :, 0] = color[2]
            bg[:, :, 1] = color[1]
            bg[:, :, 2] = color[0]

            pred_mask = cv2.warpAffine(pred_mask, invM, (img_width, img_height), flags=cv2.INTER_LINEAR)
            matts = 0.5 * pred_mask[:, :, None]
            draw_img = draw_img * (1.0 - matts) + bg * matts

        draw_img = draw_img.astype(np.uint8)
        out_path = os.path.join(args.output_dir, "pred_mask.png")

        if not os.path.exists(out_path):
            cv2.imwrite(out_path, draw_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--config-file', required=True, type=str)
    parser.add_argument('--img_path', required=True, type=str)
    parser.add_argument('--text_prompts', required=True, type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--scoreThres', default=0.3, type=float)
    parser.add_argument('--iouThres', default=0.55, type=float)

    args = parser.parse_args()

    inference = Inference(args, 'cuda:0')
    inference.predict_with_args(args)
