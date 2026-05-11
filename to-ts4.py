#!/usr/bin/env python
# to-ts_my3.py : .pth -> TorchScript(.ts) converter (UNet / UResNet / NestedUNet / MobileNetV2 / MobileNetV3 / Transformer_UNet)

import argparse
import os
import sys
import numpy as np
import torch

from unet import UNet
from uresnet import UResNet
from nestedunet import NestedUNet
from mobilenetv2.model import MobileNet_UNet
from mobilenetv3.model import MobileNetV3_UNet
from transformer.model import Transformer_UNet


def get_args():
    p = argparse.ArgumentParser(description="Convert a PyTorch .pth to TorchScript .ts")
    p.add_argument('--model', '-m', required=True, help="Path to .pth checkpoint")
    p.add_argument('--arch', '-a',
                   choices=['unet', 'uresnet', 'nestedunet', 'mobilenetv2', 'mobilenetv3', 'transformer'],
                   default='unet', help="Model architecture")
    p.add_argument('--gpu', '-g', action='store_true', help="Use CUDA (default: CPU)")
    p.add_argument('--output', '-o', default=None, help="Output .ts path (default: <pth_name>_<arch>.ts)")

    # I/O shape
    p.add_argument('--input-ch', type=int, default=3, help="Input channels")
    p.add_argument('--output-ch', type=int, default=1, help="Output channels")
    p.add_argument('--height', type=int, default=800, help="Dummy input height for tracing")
    p.add_argument('--width', type=int, default=600, help="Dummy input width for tracing")

    # MobileNetV3 options
    p.add_argument('--mv3-variant', choices=['large', 'small'], default='large',
                   help="MobileNetV3 variant")
    p.add_argument('--mv3-pretrained', action='store_true',
                   help="Instantiate MobileNetV3_UNet with pretrained=True (usually unnecessary for conversion)")

    # Transformer options
    p.add_argument('--tr-embed-dim', type=int, default=512, help="Transformer embed dim")
    p.add_argument('--tr-heads', type=int, default=8, help="Transformer #heads")
    p.add_argument('--tr-depth', type=int, default=4, help="Transformer depth (#blocks)")
    p.add_argument('--tr-mlp-ratio', type=float, default=4.0, help="Transformer MLP ratio")
    p.add_argument('--tr-drop', type=float, default=0.0, help="Transformer dropout")
    p.add_argument('--tr-attn-drop', type=float, default=0.0, help="Transformer attn dropout")

    return p.parse_args()


def count_params(net):
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'params = {params:,}')


def strip_module_prefix(state_dict):
    # handle DataParallel checkpoints with 'module.' prefix
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace('module.', '', 1)] = v
    return new_sd


def build_model(args):
    ic, oc = args.input_ch, args.output_ch
    if args.arch == 'unet':
        return UNet(ic, oc)
    elif args.arch == 'uresnet':
        return UResNet(ic, oc)
    elif args.arch == 'nestedunet':
        return NestedUNet(ic, oc)
    elif args.arch == 'mobilenetv2':
        return MobileNet_UNet(ic, oc)
    elif args.arch == 'mobilenetv3':
        return MobileNetV3_UNet(ic, oc, variant=args.mv3_variant, pretrained=args.mv3_pretrained)
    elif args.arch == 'transformer':
        return Transformer_UNet(
            input_channels=ic, output_channels=oc,
            embed_dim=args.tr_embed_dim, num_heads=args.tr_heads, depth=args.tr_depth,
            mlp_ratio=args.tr_mlp_ratio, drop=args.tr_drop, attn_drop=args.tr_attn_drop
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")


def safe_load_state_dict(net, ckpt_path, device):
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(ckpt_path, map_location=device)

    if isinstance(obj, dict):
        if 'state_dict' in obj:
            sd = obj['state_dict']
        else:
            sd = obj
    else:
        raise RuntimeError(f"Unexpected checkpoint type: {type(obj)}")

    sd = strip_module_prefix(sd)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing first 5) -> {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing first 5) -> {unexpected[:5]}")


def try_script_then_trace(net, example, use_cuda):
    net.eval()
    try:
        print("Trying torch.jit.script ...")
        sm = torch.jit.script(net)
        with torch.no_grad():
            _ = sm(example.cuda() if use_cuda else example)
        print("Success: torch.jit.script")
        return sm
    except Exception as e:
        print(f"[script failed] {e}")

    try:
        print("Falling back to torch.jit.trace ...")
        with torch.no_grad():
            sm = torch.jit.trace(net, example.cuda() if use_cuda else example, strict=False)
            _ = sm(example.cuda() if use_cuda else example)
        print("Success: torch.jit.trace")
        return sm
    except Exception as e:
        print(f"[trace failed] {e}")
        raise RuntimeError("Both script and trace failed.") from e


def main():
    args = get_args()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    use_cuda = (device.type == 'cuda')
    print(f"Device: {device}")

    net = build_model(args)
    net.to(device)
    count_params(net)

    safe_load_state_dict(net, args.model, device)

    example = torch.rand(1, args.input_ch, args.height, args.width, device=device)

    sm = try_script_then_trace(net, example, use_cuda)

    out_ts = args.output or (os.path.splitext(os.path.basename(args.model))[0] + f"_{args.arch}.ts")
    sm.save(out_ts)
    print(f"Saved TorchScript model to: {out_ts}")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

