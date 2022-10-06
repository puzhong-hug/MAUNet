# -*- coding: utf-8 -*-
# @File    : DCT.py
# @Software: PyCharm
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


logger = logging.getLogger(__name__)

class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    def __init__(self, config, vis,channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()

        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False)
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])



    def forward(self, emb1,emb2, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
        else: weights=None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None

        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None

        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None

        return O1,O2, weights



class Mlp(nn.Module):
    def __init__(self,config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.attn_norm =  LayerNorm(config.KV_size,eps=1e-6)
        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        # self.ffn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        # self.ffn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.ffn1 = Mlp(config,channel_num[0],channel_num[0]*expand_ratio)
        self.ffn2 = Mlp(config,channel_num[1],channel_num[1]*expand_ratio)


    def forward(self, emb1,emb2,):
        embcat = []
        org1 = emb1
        org2 = emb2

        for i in range(2):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None

        emb_all = self.attn_norm(emb_all)
        cx1,cx2, weights = self.channel_attn(cx1,cx2,emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None

        org1 = cx1
        org2 = cx2
        # org3 = cx3
        # org4 = cx4
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None

        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None

        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None

        return x1, x2, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)

        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2):
        attn_weights = []
        for layer_block in self.layer:
            emb1,emb2, weights = layer_block(emb1,emb2)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None

        return emb1,emb2, attn_weights


class DCT13(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 256], patchSize=[32, 8]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]

        self.embeddings_1 = Channel_Embeddings(config,self.patchSize_1, img_size=img_size,    in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config,self.patchSize_2, img_size=img_size//4, in_channels=channel_num[1])

        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,scale_factor=(self.patchSize_1,self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2))

    def forward(self,en1,en2):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)

        encoded1, encoded2, attn_weights = self.encoder(emb1,emb2)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None


        x1 = x1 + en1  if en1 is not None else None
        x2 = x2 + en2  if en2 is not None else None


        return x1, x2, attn_weights

