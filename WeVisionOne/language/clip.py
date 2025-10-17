import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from transformers import (AutoTokenizer, CLIPTextConfig)
from transformers import CLIPTextModelWithProjection as CLIPTP


@BACKBONE_REGISTRY.register()
class CLIPLangEncoder(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()

        text_cfg = cfg.MODEL.TEXT

        model_name = text_cfg.MODEL_NAME
        token_name = text_cfg.TOKEN_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(token_name)
        clip_config = CLIPTextConfig.from_pretrained(model_name)
                                                     # attention_dropout=dropout)
        if 'CLIP-ViT-L-14-laion2B-s32B-b82K' in model_name:
            clip_config.projection_dim = 768
        
        # self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self.model = CLIPTP(config=clip_config)

    def forward(self, text):
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(num_per_batch), (
            'number of sequences not equal in batch')
        text = list(itertools.chain(*text))
        text = self.tokenizer(text=text, return_tensors='pt', padding=True, max_length=20)
        text = text.to(device=self.model.device)
        txt_outputs = self.model(**text)
        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats.reshape(-1, num_per_batch[0],
                                      txt_feats.shape[-1])
        return txt_feats