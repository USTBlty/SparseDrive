from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.registry import FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.drop import build_dropout

@FEEDFORWARD_NETWORK.register_module()
class MoEFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        num_experts=4,
        num_selected_experts=2,
        init_cfg=None,
        **kwargs,
    ):
        super(MoEFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts

        if in_channels is None:
            in_channels = embed_dims
        
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        # 路由网络 (Router / Gate)
        self.gate = Linear(in_channels, num_experts)

        # 专家网络列表
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers = []
            curr_in_channels = in_channels
            for _ in range(num_fcs - 1):
                layers.append(
                    Sequential(
                        Linear(curr_in_channels, feedforward_channels),
                        self.activate,
                        nn.Dropout(ffn_drop),
                    )
                )
                curr_in_channels = feedforward_channels
            layers.append(Linear(feedforward_channels, embed_dims))
            layers.append(nn.Dropout(ffn_drop))
            self.experts.append(Sequential(*layers))

        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        
        # x shape: [bs, num_queries, embed_dims]
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])

        # 1. 路由计算
        gate_logits = self.gate(x_flat) # [total_tokens, num_experts]
        weights, selected_experts = torch.topk(gate_logits, self.num_selected_experts)
        weights = F.softmax(weights, dim=-1) # [total_tokens, num_selected_experts]

        # 2. 专家计算
        results = torch.zeros_like(x_flat)
        
        # 遍历所有专家 (虽然低效但在 Token 数量较少时更稳定，后续可优化为并行)
        for i in range(self.num_experts):
            # 找到被路由到专家 i 的 Token 索引
            batch_mask = (selected_experts == i) # [total_tokens, k]
            token_indices, k_indices = torch.nonzero(batch_mask, as_tuple=True)
            
            if token_indices.numel() > 0:
                selected_tokens = x_flat[token_indices]
                expert_out = self.experts[i](selected_tokens)
                
                # 乘以路由权重并累加
                w = weights[token_indices, k_indices].unsqueeze(-1)
                results.index_add_(0, token_indices, w * expert_out)

        out = results.reshape(original_shape)

        if not self.add_identity:
            return self.dropout_layer(out)
        
        if identity is None:
            identity = x
        
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
