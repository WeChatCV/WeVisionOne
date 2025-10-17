import torch
import torch.nn as nn
from detectron2.config import configurable

from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from ..utils.utils import MLP, gen_encoder_output_proposals, inverse_sigmoid
from .build import TRANSFORMER_DECODER_REGISTRY


@TRANSFORMER_DECODER_REGISTRY.register()
class MaskDINOLateFusion(nn.Module):
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

        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)
        self.enc_out_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

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
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.class_embed_openset = nn.Parameter(torch.empty(hidden_dim, 256))

        self.pb_embedding = nn.Embedding(2, hidden_dim)
        self.label_enc = nn.Embedding(2, hidden_dim)
        self.num_content_tokens = 1

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}

        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["num_classes"] = -1 # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
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

    def forward_prediction_heads(self, output, mask_features, input_text_embedding, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        class_embed_whole = decoder_output @ self.class_embed_openset  # FIXME use class_embed_part to represent openset projection
        match_score = class_embed_whole @ input_text_embedding.transpose(1, 2)
        outputs_mask = None
        if pred_mask == True:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return match_score, outputs_mask

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

    def pfunc(self, txt_feats):
        bs = txt_feats.shape[0]
        max_num = txt_feats.shape[1]
        pb_labels = torch.ones( (bs, max_num) )
        labels = torch.zeros_like(pb_labels).long()
        
        device = self.label_enc.weight.device
        
        m = labels.long().to(device)
        m_pb = pb_labels.long().to(device)
        box_dn_base = torch.tensor([0, 0, 1, 1]).unsqueeze(0).to(device)
        boxes = ( box_dn_base.repeat(max_num, 1) ).repeat(bs, 1, 1)

        input_label_embed = self.label_enc(m) + self.pb_embedding(m_pb)
        input_label_embed = input_label_embed + txt_feats

        input_bbox_embed = inverse_sigmoid(boxes)
        
        attn_mask = torch.ones(self.num_queries + max_num, self.num_queries + max_num)
        attn_mask[:self.num_queries, :self.num_queries] = 0
        attn_mask[self.num_queries:, self.num_queries:] = (1 - torch.eye(max_num, max_num))
        attn_mask = attn_mask.to(device).bool()

        return input_label_embed, input_bbox_embed, attn_mask

    def forward_prediction_heads_stage2(self, output, mask_features, input_text_embeding, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        outputs_mask = None
        class_embed_whole = decoder_output @ self.class_embed_openset  # FIXME use class_embed_part to represent openset projection

        match_score = None

        num_content = input_text_embeding.shape[1] # mask_dict['num_content']
        class_embed_whole_content = class_embed_whole[:, -num_content:]
        class_embed_whole_content = class_embed_whole_content.view(class_embed_whole.shape[0], -1, self.num_content_tokens, class_embed_whole.shape[-1])  # remove content embedding
        class_embed_whole_content = class_embed_whole_content[:, :, -1, :]  # select the last one of all mask tokens
    
        match_score = class_embed_whole[:, :-num_content]@class_embed_whole_content.transpose(1, 2)  # FIXME from tracking to detection
        decoder_output = decoder_output[:, :-num_content]  # remove content embedding

        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return match_score, outputs_mask


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
        
        enc_outputs_class_unselected = self.forward_prediction_heads(output_memory.transpose(0, 1), mask_features, 
                                                                     txt_feats, pred_mask=False)
        enc_outputs_class_unselected = enc_outputs_class_unselected[0]

        enc_outputs_coord_unselected = self.enc_out_bbox_embed(
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

        tgt = tgt_undetach.detach()
        if self.learn_tgt:
            tgt = self.tgt_embed.weight[None].repeat(bs, 1, 1)

        input_label_embed, input_bbox_embed, tgt_mask = self.pfunc(txt_feats)

        new_tgt = torch.cat([tgt, input_label_embed], 1)
        new_ref = torch.cat([refpoint_embed, input_bbox_embed], 1)

        hs, references = self.decoder(
            tgt=new_tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=new_ref.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )

        outputs_class, outputs_mask = self.forward_prediction_heads_stage2(hs[-1].transpose(0, 1), mask_features, txt_feats, pred_mask=True)

        out_boxes = self.pred_box(references, hs)
        outputs_bbox = out_boxes[-1]

        out = {
            'pred_logits': outputs_class,
            'pred_masks': outputs_mask,
            'pred_boxes': outputs_bbox,
        }

        return out
