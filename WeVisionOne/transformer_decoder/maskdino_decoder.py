import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from detectron2.config import configurable
from detectron2.structures import BitMasks

from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from ..utils.utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from ..utils import box_ops
from .build import TRANSFORMER_DECODER_REGISTRY


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """

    def __init__(self,
                 embed_dims: int,
                 use_einsum: bool = True) -> None:

        super().__init__()

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum('blc,bkc->blk', x, w)
        x = x * self.logit_scale.exp() + self.bias
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MaskDINODecoder(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_project: bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 4,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 4,
            dec_layer_share: bool = False,
    ):
        super().__init__()
        self.num_feature_levels = total_num_feature_levels
        self.num_layers = dec_layers
        self.num_queries = num_queries
        self.learn_tgt = learn_tgt
        self.num_classes = num_classes

        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          )
        self.hidden_dim = hidden_dim
        self._bbox_embed = _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        box_embed_layerlist = [_bbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.bbox_embed = self.bbox_embed

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        if learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.cls_constrasts = ContrastiveHead(embed_dims=num_classes)
        self.initialize_box_type = 'bitmask'


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}

        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.TRANSFORMER_DECODER.HIDDEN_DIM

        ret["num_queries"] = cfg.MODEL.TRANSFORMER_DECODER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.TRANSFORMER_DECODER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.TRANSFORMER_DECODER.DIM_FEEDFORWARD
        ret["dec_layers"] = cfg.MODEL.TRANSFORMER_DECODER.DEC_LAYERS
        ret["enforce_input_project"] = False # cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.TRANSFORMER_DECODER.MASK_DIM
        ret["learn_tgt"] = cfg.MODEL.TRANSFORMER_DECODER.LEARN_TGT
        ret["total_num_feature_levels"] = cfg.MODEL.TRANSFORMER_DECODER.TOTAL_NUM_FEATURE_LEVELS
        return ret

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward_prediction_heads(self, output, mask_features, txt_feats=None, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        cls_embs = self.class_embed(decoder_output)
        outputs_class = self.cls_constrasts(cls_embs, txt_feats)

        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = reference[0].device

        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs).to(device)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, masks, txt_feats=None):
        size_list = []
        src_flatten = []
        spatial_shapes = []
        mask_flatten = []

        if masks is None:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]        

        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs, c, h, w = x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            flatten = x[idx].flatten(2)
            src_flatten.append(flatten.transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))

        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)        

        output_memory, output_proposals = gen_encoder_output_proposals(
            src_flatten, mask_flatten, spatial_shapes
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        
        enc_outputs_class_unselected, _ = self.forward_prediction_heads(output_memory.transpose(0, 1), mask_features, 
                                                                     txt_feats, pred_mask=False)

        enc_outputs_coord_unselected = self._bbox_embed(
            output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
        topk = self.num_queries
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoid
        refpoint_embed = refpoint_embed_undetach.detach()

        # gather tgt
        tgt_undetach = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        )

        _, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features, txt_feats=txt_feats, pred_mask=True)

        tgt = tgt_undetach.detach()
        if self.learn_tgt:
            tgt = self.tgt_embed.weight[None].repeat(bs, 1, 1)

        if self.initialize_box_type != 'no':
            device = src_flatten.device
            flaten_mask = outputs_mask.detach().flatten(0, 1)
            h, w = outputs_mask.shape[-2:]
            if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
            elif self.initialize_box_type == 'mask2box':  # faster conversion
                refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
            else:
                assert NotImplementedError
            refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],
                                                                                          dtype=torch.float).to(device)
            refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
            refpoint_embed = inverse_sigmoid(refpoint_embed)

        tgt_mask = None
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        outputs_class, outputs_mask = self.forward_prediction_heads(hs[-1].transpose(0, 1), mask_features, txt_feats, pred_mask=True)

        out_boxes = self.pred_box(references, hs)
        outputs_bbox = out_boxes[-1]

        out = {
            'pred_logits': outputs_class,
            'pred_masks': outputs_mask,
            'pred_boxes': outputs_bbox,
        }

        return out
