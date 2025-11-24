import torch
import torch.nn as nn

from .ltae import LTAE2d
from typing import Dict, Any
from utils.const import FIX_PADDING_MODE


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, block_config, norm, **norm_params):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            # fixed padding method for conv
            nn.Conv2d(in_channel, out_channel, padding_mode=FIX_PADDING_MODE, **block_config),
            # group=4 is serve for hidden-channel, only "batch", "group" mode
            nn.BatchNorm2d(out_channel) if norm == "batch" else nn.GroupNorm(num_channels=out_channel, **norm_params),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        if x.ndim == 4:
            return self.block(x)
        elif x.ndim == 5:
            b, t, c, h, w = x.shape
            out = x.view(b * t, c, h, w)
            out = self.block(out)
            _, c2, h2, w2 = out.shape
            out = out.view(b, t, c2, h2, w2)
            return out
        else:
            raise NotImplementedError(f"ndim {x.ndim} is not supported")

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_block_config, ch_block_config ,norm, **norm_params):
        super(DownBlock, self).__init__()

        self.down = ConvBlock(in_channel, out_channel=in_channel, block_config=down_block_config, norm=norm, **norm_params)
        self.conv1 = ConvBlock(in_channel, out_channel=out_channel, block_config=ch_block_config, norm=norm, **norm_params)
        self.conv2 = ConvBlock(out_channel, out_channel=out_channel, block_config=ch_block_config, norm=norm, **norm_params)

    def forward(self, x):
        out = self.down(x)
        out = self.conv1(out)
        out = out + self.conv2(out)     # res block
        return out

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, skip_channel, up_block_config, ch_block_config, norm, **norm_params):
        super(UpBlock, self).__init__()

        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channel, skip_channel, kernel_size=1),    # 1x1 Conv2d, filter channel
            nn.BatchNorm2d(skip_channel),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, **up_block_config),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.conv1 = ConvBlock(out_channel + skip_channel, out_channel, block_config=ch_block_config, norm=norm, **norm_params)
        self.conv2 = ConvBlock(out_channel, out_channel, block_config=ch_block_config, norm=norm, **norm_params)

    def forward(self, x, skip):
        out = self.up(x)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out

class TemporalAggregator(nn.Module):
    def __init__(self):
        super(TemporalAggregator, self).__init__()
        self.mode = "att_group"

    def forward(self, x, attn_mask):
        assert self.mode == "att_group"

        n_heads, batch, time, h_att, w_att = attn_mask.shape
        _, _, c, h_x, w_x = x.shape

        # flatten heads
        attn = attn_mask.view(n_heads * batch, time, h_att, w_att)

        if h_x > h_att:
            attn = nn.Upsample(size=(h_x, w_x), mode="bilinear", align_corners=False)(attn)
        else:
            attn = nn.AvgPool2d(kernel_size=w_att // w_x)(attn)

        attn = attn.view(n_heads, batch, time, h_x, w_x)

        x_groups = x.chunk(n_heads, dim=2)                      # (B,T,C_head,H,W)
        x_stack = torch.stack(x_groups)                         # (Head,B,T,C_head,H,W)

        weighted = attn[:, :, :, None, :, :] * x_stack          # (Head,B,T,C_head,H,W)
        out = weighted.sum(dim=2)                               # (Head,B,C_head,H,W)

        out = torch.cat([head_out for head_out in out], dim=1)  # (B,C_cls,H,W)
        return out


class UTAE(nn.Module):
    def __init__(
            self,
            config: Dict[str, Any],
    ):
        super(UTAE, self).__init__()
        # validate
        # ...

        norm = config["norm"]
        in_channel = config["in_channel"]
        norm_params = config["norm_params"]
        enc_channels = config["enc_channels"]
        dec_channels = config["dec_channels"]
        out_channel = config["out_channel"]

        self.config = config
        self.level = len(enc_channels)
        self.return_decs = config["is_return_decs"]
        self.return_attn = config["is_return_attn"]

        # input part
        self.input_block = nn.Sequential(
            ConvBlock(in_channel, enc_channels[0], block_config=self.config["input_block"], norm=norm, **norm_params),
            ConvBlock(enc_channels[0], enc_channels[0], block_config=self.config["input_block"], norm=norm, **norm_params),
        )   # self.input_block(x) | forward

        # u-net encoder block-list
        self.down_blocks = nn.ModuleList(
            DownBlock(
                in_channel=enc_channels[i],
                out_channel=enc_channels[i + 1],
                down_block_config=self.config["enc_down_block"],
                ch_block_config=self.config["enc_ch_block"],
                norm=norm,
                **norm_params,
            )
            for i in range(self.level - 1)
        )
        # u-net decoder
        self.up_blocks = nn.ModuleList(
            UpBlock(
                in_channel=dec_channels[i],
                out_channel=dec_channels[i + 1],
                skip_channel=enc_channels[-(i + 2)],
                up_block_config=self.config["dec_up_block"],
                ch_block_config=self.config["dec_ch_block"],
                norm=norm,
                **norm_params,
            )
            for i in range(self.level - 1)
        )

        # temporal encoder
        self.temporal_block = LTAE2d(in_channel=enc_channels[-1], out_channel=dec_channels[0], config=config["ltae"])
        self.temporal_aggregator = TemporalAggregator()

        # output part
        self.output_block = nn.Sequential(
            ConvBlock(in_channel=dec_channels[-1], out_channel=dec_channels[-1], block_config=self.config["out_block"], norm=norm, **norm_params),
            ConvBlock(in_channel=dec_channels[-1], out_channel=out_channel, block_config=self.config["out_block"], norm="batch", **norm_params),
        )

    def forward(self, inputs):
        # x shape like: (B,T,C,H,W)
        # doys shape like: (B,T)
        # specific settings for PoyangDatasets
        x, doy_vec = inputs["s2"], inputs["s2_doy"]

        enc_l1 = self.input_block(x)

        # save different level encoder feature
        enc_list = [enc_l1]

        for i in range(self.level - 1):
            enc_li = self.down_blocks[i](enc_list[-1])
            enc_list.append(enc_li)

        dec_le, attn = self.temporal_block(x=enc_list[-1], pos_vec=doy_vec)
        dec_list = [dec_le] if self.return_decs else None

        dec_li = dec_le
        for i in range(self.level - 1):
            skip = self.temporal_aggregator(
                # last enc_out do not need temporal_aggregator, u-shape iteration start with -2
                enc_list[-(i + 2)], attn_mask=attn
            )
            dec_li = self.up_blocks[i](dec_li, skip)
            if self.return_decs:
                dec_list.append(dec_li)
        dec_l1 = dec_li

        # return out_only | with_attn | with_decs
        if self.return_decs:
            return dec_l1, dec_list
        elif self.return_attn:
            return dec_l1, attn
        else:
            return dec_l1