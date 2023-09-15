# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm,BatchNorm1d
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs

from .modeling_resnet import ResNetV2
import torch.nn.utils.weight_norm as weightNorm

from torch.nn import functional as F
from abc import abstractmethod

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
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


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, bottleneck=768, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.bottleneck = bottleneck
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)

        ## 提取到的特征
        self.bottle = nn.Linear(config.hidden_size, self.bottleneck)
        self.normalazation = LayerNorm(self.bottleneck, eps=1e-6)

        ## Attention mask
        self.em = nn.Embedding(2, self.bottleneck)
        self.mask = torch.empty(1, self.bottleneck)

        # self.head = Linear(config.hidden_size, num_classes)
        # self.head = weightNorm(nn.Linear(self.bottleneck, num_classes), name="weight")
        self.head = nn.Linear(self.bottleneck, num_classes)

    def netF(self, x):
        x, attn_weights = self.transformer(x)
        x = self.bottle(x[:, 0])
        x = self.normalazation(x)
        return x

    def netB(self, x, s=100, t=0, all_out=False):
        '''
        将生成的特征和相应域的mask相乘，防止遗忘
        return:源域训练放回全部输出和mask
            目标域训练返回输出和当前mask，分为域已知训练和域未知训练，细分为软标签计算mask和硬标签计算mask
        '''
        t0 = torch.LongTensor([0]).cuda()
        t1 = torch.LongTensor([1]).cuda()
        mask0 = nn.Sigmoid()(self.em(t0) * s)
        mask1 = nn.Sigmoid()(self.em(t1) * s)
        mask_bank = torch.randn(2, self.bottleneck).cuda()
        mask_bank[0] = mask0
        mask_bank[1] = mask1
        if all_out:
            self.mask = mask0
            out0 = x * mask0
            out1 = x * mask1
            return (out0, out1), (self.mask, mask1)
        else:
            out = x * mask_bank[t]
            # print(out.shape)
            return out, mask_bank[t]

    def netC(self, x):
        x = self.head(x)
        return x

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


class VisionTransformer_cda(nn.Module):
    def __init__(self, config, img_size=256, num_classes=21843, bottleneck=768 * 2, zero_head=False, vis=False):
        super(VisionTransformer_cda, self).__init__()
        self.bottleneck = bottleneck
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)

        ## 提取到的特征
        self.bottle = nn.Linear(config.hidden_size, self.bottleneck)
        self.normalazation = LayerNorm(self.bottleneck, eps=1e-6)

        ## Attention mask
        self.em = nn.Embedding(4, self.bottleneck)
        self.mask = torch.empty(1, self.bottleneck)

        # self.head = Linear(config.hidden_size, num_classes)
        self.head = weightNorm(nn.Linear(self.bottleneck, num_classes), name="weight")
        # self.head = Linear(config.hidden_size, num_classes)

    def netF(self, x):
        x, attn_weights = self.transformer(x)
        x = self.bottle(x[:, 0])
        x = self.normalazation(x)
        return x

    def netB(self, x, s=100, t=0, all_out=False):
        '''
        将生成的特征和相应域的mask相乘，防止遗忘
        return:源域训练放回全部输出和mask
            目标域训练返回输出和当前mask，分为域已知训练和域未知训练，细分为软标签计算mask和硬标签计算mask
        '''
        t0 = torch.LongTensor([0]).cuda()
        t1 = torch.LongTensor([1]).cuda()
        t2 = torch.LongTensor([2]).cuda()
        t3 = torch.LongTensor([3]).cuda()
        mask0 = nn.Sigmoid()(self.em(t0) * s)
        mask1 = nn.Sigmoid()(self.em(t1) * s)
        mask2 = nn.Sigmoid()(self.em(t2) * s)
        mask3 = nn.Sigmoid()(self.em(t3) * s)
        mask_bank = torch.randn(4, self.bottleneck).cuda()
        mask_bank[0] = mask0
        mask_bank[1] = mask1
        mask_bank[2] = mask2
        mask_bank[3] = mask3
        if all_out:
            self.mask = mask0
            out0 = x * mask0
            out1 = x * mask1
            out2 = x * mask2
            out3 = x * mask3
            return (out0, out1, out2, out3), (self.mask, mask1, mask2, mask3)
        else:
            out = x * mask_bank[t]
            # print(out.shape)
            return out, mask_bank[t]

    def netC(self, x):
        x = self.head(x)
        return x

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


class VisionTransformer_DomainClassifier(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False,
                 vis=False, bottleneck=256, domain_num=1, freeze_layers=6):
        super(VisionTransformer_DomainClassifier, self).__init__()
        self.bottleneck = bottleneck
        self.domain_num = domain_num
        self.freeze_layers = freeze_layers
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)

        ## 固定前6block送进域判别器
        self.bot_cls = nn.Linear(config.hidden_size, self.bottleneck)
        self.domain_classifier = nn.Sequential(nn.Linear(self.bottleneck, 64), nn.ReLU(), nn.Linear(64, self.domain_num))

        ## 提取到的特征
        self.bottle = nn.Linear(config.hidden_size, self.bottleneck)
        self.normalazation = LayerNorm(self.bottleneck, eps=1e-6)

        ## Attention mask
        self.em = nn.Embedding(2, self.bottleneck)
        self.mask = torch.empty(1, self.bottleneck)

        # self.head = Linear(config.hidden_size, num_classes)
        self.head = weightNorm(nn.Linear(self.bottleneck, num_classes), name="weight")

    def freeze_netF(self, x):
        hidden_states = self.transformer.embeddings(x)
        # print(self.transformer.encoder.layer[:self.freeze_layers])
        for layer_block in self.transformer.encoder.layer[:self.freeze_layers]:
            hidden_states, weights = layer_block(hidden_states)
        encoded = self.transformer.encoder.encoder_norm(hidden_states)
        # x = self.transformer.encoder.layer[:self.freeze_layers](embedding_output)
        x = self.bot_cls(encoded[:, 0])  ##映射到bottleneck（256）维
        return x

    def netD(self, x, args, netG=None, test=False):
        ''''
        域判别器网络
        return:判别器训练输出，训练数据真实标签，当前任务数据预测输出
        '''
        if test == True:
            out = self.domain_classifier(x)
            return out
        with torch.no_grad():
            # 随机从隐变量的分布中取隐变量
            z = torch.randn(args.batch_size, args.z_dim).cuda()  # 每一行是一个隐变量，总共有batch_size行
            c = np.random.randint(0, self.num_classes, size=z.shape[0])
            c = torch.FloatTensor(c)
            random_res = netG.decode(z, c)
            x_generated = random_res[:, 0:random_res.shape[1] - self.num_classes]
        domain_label1 = torch.ones(x.shape[0], dtype=torch.int64)
        domain_label2 = torch.zeros(x_generated.shape[0], dtype=torch.int64)
        feature_data = torch.cat((x, x_generated))
        domain_label = torch.cat((domain_label1, domain_label2))
        domain_label = torch.eye(2)[domain_label].cuda()

        # inputs_np = feature_data.detach().cpu().numpy()
        # labels_np = domain_label.detach().cpu().numpy()
        # state = np.random.get_state()
        # np.random.shuffle(inputs_np)
        # np.random.set_state(state)
        # np.random.shuffle(labels_np)

        # inputs = torch.from_numpy(inputs_np[:64]).cuda()
        # labels = torch.from_numpy(labels_np[:64]).cuda().long()

        out = self.domain_classifier(feature_data)
        return out, domain_label, out[0:x.shape[0]]

    def netF(self, x):
        x, attn_weights = self.transformer(x)
        x = self.bottle(x[:, 0])
        x = self.normalazation(x)
        return x

    def netB(self, x, s=100, t=0, all_out=False, d_cls_out=None, type=None):
        '''
        将生成的特征和相应域的mask相乘，防止遗忘
        return:源域训练放回全部输出和mask
            目标域训练返回输出和当前mask，分为域已知训练和域未知训练，细分为软标签计算mask和硬标签计算mask
        '''
        t0 = torch.LongTensor([0]).cuda()
        t1 = torch.LongTensor([1]).cuda()
        mask0 = nn.Sigmoid()(self.em(t0) * s)
        mask1 = nn.Sigmoid()(self.em(t1) * s)
        mask_bank = torch.randn(2, self.bottleneck).cuda()
        mask_bank[0] = mask0
        mask_bank[1] = mask1
        if all_out:
            self.mask = mask0
            out0 = x * mask0
            out1 = x * mask1
            return (out0, out1), (self.mask, mask1)
        elif type == 'soft_mask':
            masks = torch.cat([mask0, mask1], dim=0)
            # print(masks.shape)
            # print(x.shape)
            d_cls_out = nn.Softmax(dim=1)(d_cls_out)
            # print(d_cls_out.shape)
            soft_mask = torch.mm(d_cls_out, masks)
            # print(soft_mask.shape)
            return x * soft_mask, soft_mask
        elif type == 'hard_mask':
            _, pred_domain = torch.max(d_cls_out, 1)
            hard_mask = mask_bank[pred_domain]
            return x * hard_mask, hard_mask
        else:
            out = x * mask_bank[t]
            # print(out.shape)
            return out, mask_bank[t]

    def netC(self, x):
        x = self.head(x)
        return x

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


class VisionTransformer_DomainClassifier_cda(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False,
                 vis=False, bottleneck=256, domain_num=1, freeze_layers=6):
        super(VisionTransformer_DomainClassifier_cda, self).__init__()
        self.bottleneck = bottleneck
        self.domain_num = domain_num
        self.freeze_layers = freeze_layers
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)

        ## 固定前6block送进域判别器
        self.bot_cls = nn.Linear(config.hidden_size, self.bottleneck)
        self.domain_classifier = nn.Sequential(LayerNorm(self.bottleneck, eps=1e-6), nn.Linear(self.bottleneck, 128),
                                        LayerNorm(128, eps=1e-6),
                                        nn.Linear(128, domain_num))
        self.VAEnormalazation = LayerNorm(self.bottleneck, eps=1e-6)
        ## 提取到的特征
        self.bottle = nn.Linear(config.hidden_size, self.bottleneck)
        self.normalazation = LayerNorm(self.bottleneck, eps=1e-6)

        ## Attention mask
        self.em = nn.Embedding(4, self.bottleneck)
        self.mask = torch.empty(1, self.bottleneck)

        # self.head = Linear(config.hidden_size, num_classes)
        self.head = weightNorm(nn.Linear(self.bottleneck, num_classes), name="weight")

    def freeze_netF(self, x):
        with torch.no_grad():
            hidden_states = self.transformer.embeddings(x)
            for layer_block in self.transformer.encoder.layer[:self.freeze_layers]:
                hidden_states, weights = layer_block(hidden_states)
        # encoded = self.transformer.encoder.encoder_norm(hidden_states) ##有问题
        # x = self.transformer.encoder.layer[:self.freeze_layers](embedding_output)
        # x = self.bot_cls(encoded[:, 0])  ##映射到bottleneck（256）维
        x = self.VAEnormalazation(hidden_states[:, 0])
        return x

    def CVAE_features(self, x):
        x = self.transformer.embeddings(x)
        for i, layer_block in enumerate(self.transformer.encoder.layer):
            if i > 0:
                break
            x, _ = layer_block(x)
        return x[:, 0], x

    def netD(self, x, args, task_pre=None, t=None, netG=None, test=False):
        ''''
        域判别器网络
        return:判别器训练输出，训练数据真实标签，当前任务数据预测输出
        '''
        if test == True:
            out = self.domain_classifier(x)
            return out
        with torch.no_grad():
            # 随机从隐变量的分布中取隐变量
            z = torch.randn(x.shape[0], args.z_dim).cuda()  # 每一行是一个隐变量，总共有batch_size行
            # c = np.random.randint(0, self.num_classes, size=z.shape[0])
            # c = torch.FloatTensor(c)
            x_generated = netG.decode(z).view(-1, 768)
        domain_label_t = torch.ones(x.shape[0], dtype=torch.int64) * t
        domain_label_pre = torch.ones(x_generated.shape[0], dtype=torch.int64) * task_pre
        feature_data = torch.cat((x, x_generated))
        domain_label = torch.cat((domain_label_t, domain_label_pre))
        domain_label = torch.eye(4)[domain_label].cuda()

        out = self.domain_classifier(feature_data)
        return out, domain_label, out[0:x.shape[0]]

    def netD_(self, x, t=None, netG=None, test=False):

        ''''
        域判别器网络
        return:判别器训练输出，训练数据真实标签，当前任务数据预测输出
        '''
        import random
        if test == True:
            out = self.domain_classifier(x)
            return out
        with torch.no_grad():
            domain_label = torch.ones(x.shape[0], dtype=torch.int64) * t
            feature_data = x
            for t_pre in range(t):
                z = torch.randn(x.shape[0], netG[t_pre].z_dim).cuda()
                c = np.random.randint(0, netG[t_pre].y_dim, size=x.shape[0])
                c = torch.FloatTensor(c)
                random_res = netG[t_pre].decode(z, c)
                x_pre = random_res[:, 0:random_res.shape[1] - netG[t_pre].y_dim]
                domain_label_pre = torch.ones(x_pre.shape[0], dtype=torch.int64) * t_pre
                domain_label = torch.cat((domain_label, domain_label_pre))
                feature_data = torch.cat((feature_data, x_pre))
            domain_label = torch.eye(4)[domain_label].cuda()

        out = self.domain_classifier(feature_data)
        return out, domain_label, out[0:x.shape[0]]

    def netF(self, x):
        x, attn_weights = self.transformer(x)
        x = self.bottle(x[:, 0])
        x = self.normalazation(x)
        return x

    def netF_(self, x):
        for i, layer_block in enumerate(self.transformer.encoder.layer):
            if i == 0:
                continue
            x, _ = layer_block(x)
        x=self.transformer.encoder.encoder_norm(x)
        x = self.bottle(x[:, 0])
        x = self.normalazation(x)
        return x

    def netB(self, x, s=100, t=0, all_out=False, d_cls_out=None, type=None):
        '''
        将生成的特征和相应域的mask相乘，防止遗忘
        return:源域训练放回全部输出和mask
            目标域训练返回输出和当前mask，分为域已知训练和域未知训练，细分为软标签计算mask和硬标签计算mask
        '''
        t0 = torch.LongTensor([0]).cuda()
        t1 = torch.LongTensor([1]).cuda()
        t2 = torch.LongTensor([2]).cuda()
        t3 = torch.LongTensor([3]).cuda()
        mask0 = nn.Sigmoid()(self.em(t0) * s)
        mask1 = nn.Sigmoid()(self.em(t1) * s)
        mask2 = nn.Sigmoid()(self.em(t2) * s)
        mask3 = nn.Sigmoid()(self.em(t3) * s)
        # mask3 = nn.Sigmoid()(torch.randn(1, self.bottleneck).cuda()*0)
        mask_bank = torch.randn(4, self.bottleneck).cuda()
        mask_bank[0] = mask0
        mask_bank[1] = mask1
        mask_bank[2] = mask2
        mask_bank[3] = mask3
        if all_out:
            self.mask = mask0
            out0 = x * mask0
            out1 = x * mask1
            out2 = x * mask2
            out3 = x * mask3
            return (out0, out1, out2, out3), (self.mask, mask1, mask2, mask3)
        elif type == 'soft_mask':
            if d_cls_out.shape[1] == 3:
                masks = torch.cat([mask0, mask1, mask2], dim=0)
            elif d_cls_out.shape[1]== 4:
                masks = torch.cat([mask0, mask1, mask2, mask3], dim=0)
            else:
                masks = torch.cat([mask0, mask1], dim=0)
            d_cls_out = nn.Softmax(dim=1)(d_cls_out)
            soft_mask = torch.mm(d_cls_out, masks)
            return x * soft_mask, soft_mask
        elif type == 'hard_mask':
            _, pred_domain = torch.max(d_cls_out, 1)
            hard_mask = mask_bank[pred_domain]
            return x * hard_mask, hard_mask
        else:
            out = x * mask_bank[t]
            # print(out.shape)
            return out, mask_bank[t]

    def netC(self, x):
        x = self.head(x)
        return x

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs):
        raise NotImplementedError

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass


class VAE(BaseVAE):

    def __init__(self, in_channels=3,
                 latent_dim=20,
                 hidden_dims=None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 128, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        # print(input.shape)
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
