import torch
import numpy as np
import torch.nn as nn

from typing import Any, Dict


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, scale=1000, repeat=None, offset=0):
        super(PositionalEncoder, self).__init__()
        self.scale = scale
        self.d_model = d_model
        self.repeat = repeat
        self.denom = torch.pow(
            scale, 2 * (torch.arange(offset, offset + d_model).float() // 2) / d_model
        )
        self.updated_location = False

    # (B,T) -> (B,T,d_model)
    def forward(self, batch_positions):
        # load at device only once
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # (B,T,C)
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        # Output shape like: (Head, B, d_model // Head)
        # d_v is computed from the head tensor d_model,
        # but the channel dimension of the intermediate
        # tensor must be manually set to d_k.
        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class LTAE2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            config: Dict[str, Any],
    ):
        super(LTAE2d, self).__init__()

        # Usage:
        #   Input: (N,C, *) where C=num_channels
        #   Output: (N,C, *) (same shape as input)
        # The default batch dimension is placed at the forefront
        self.return_attn = config["is_return_attn"]
        self.d_model = config["d_model"]
        self.n_head = config["n_head"]
        self.d_k = config["d_k"]

        mlp_dim = config["mlp"]
        pe_scale = config["pe_scale"]
        dropout = config["dropout"]

        self.input_norm = nn.GroupNorm(
            num_groups=self.n_head,
            num_channels=in_channel
        )
        self.output_norm = nn.GroupNorm(
            num_groups=self.n_head,
            num_channels=out_channel
        )

        # 1x1 Conv2d to expand feature dimension to d_model
        self.input_conv = nn.Conv1d(in_channel, self.d_model, kernel_size=1)

        # position encoder
        self.pos_encoder = PositionalEncoder(self.d_model // self.n_head, scale=pe_scale, repeat=self.n_head)
        self.attention_heads = MultiHeadAttention(n_head=self.n_head, d_k=self.d_k, d_in=self.d_model)

        layers = []
        for i in range(len(mlp_dim) - 1):
            layers.extend([
                nn.Linear(mlp_dim[i], mlp_dim[i + 1]),
                nn.BatchNorm1d(mlp_dim[i + 1]),
                nn.ReLU(),
            ])

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, pos_vec):
        batch, time, ch, h, w = x.shape

        # (B,T,C,H,W) -> (N,T,C), note "BHW" as "N"
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(batch * h * w, time, ch)
        # compute (N,C,T), computation of GN is independent across batches.
        out = self.input_norm(out.permute(0, 2, 1))
        out = self.input_conv(out)

        # reshape to (N,T,C)
        out = out.permute(0, 2, 1)

        # position encoding
        pos = (
            pos_vec.unsqueeze(-1)
            .repeat((1, 1, h))
            .unsqueeze(-1)
            .repeat((1, 1, 1, w))
        )   # (B,T,H,W)
        pos = pos.permute(0, 2, 3, 1).contiguous().view(batch * h * w, time)  # (N,T)

        out = out + self.pos_encoder(batch_positions=pos)
        out, attn = self.attention_heads(out)
        out = out.permute(1, 0, 2).contiguous().view(batch * h * w, -1)

        out = self.dropout(self.mlp(out))
        out = self.output_norm(out)
        out = out.view(batch, h, w, -1).permute(0, 3, 1, 2)

        # (Head,B,T,H,W)
        attn = attn.view(self.n_head, batch, h, w, time).permute(0, 1, 4, 2, 3)

        if self.return_attn:
            return out, attn
        else:
            return out