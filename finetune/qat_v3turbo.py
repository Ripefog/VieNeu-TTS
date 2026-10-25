"""Quantization-aware training (QAT) to int8 for VieNeu-TTS v3 Turbo.

The CPU engine (``OnnxV3LiteEngine``, ``precision="int8"``) runs graphs produced by
onnxruntime's dynamic quantization with this exact recipe (verified bitwise against
the shipped ``onnx_int8``):

    quantize_dynamic(model, out, op_types_to_quantize=["MatMul"],
                     per_channel=True, weight_type=QInt8, reduce_range=True)

which means per-output-channel symmetric 7-bit weights (scale = absmax/64) and
per-tensor dynamic uint8 activations, applied to exactly the 84 backbone Linear
projections (12 layers x q/k/v/o/gate/up/down) and the 5 acoustic-decoder Linears
(qkv/o_proj/ff_gate/ff_up/ff_down). Embeddings, lm heads and xvec_proj live outside
the graphs (``vieneu_v3_heads.npz``, numpy fp32) and are not quantized.

This script replicates that arithmetic as a fake-quant forward pass (straight-through
estimator), lets the Linear weights adapt to it under the same teacher-forcing loss
``train_lora.py`` uses (``vieneu_lora.compute_loss``), bakes the quantized weights
back in, and writes deployment-ready graph directories:

    <out>/<run>/onnx_qat_fp32/   fp32 backbone + freshly exported fp32 acoustic
                                 (engine precision="fp32"; ablation baseline)
    <out>/<run>/onnx_qat_int8/   both quantized with the shipped recipe
                                 (engine precision="int8"; the QAT deliverable)
    <out>/<run>/qat_model.safetensors, mapping.json, train_args.json

The backbone graphs are NOT re-exported from torch: the reference fp32 graphs
(``onnx_update``) keep their topology and only the 84 MatMul weight values are
transplanted. The anonymous ``onnx::MatMul_*`` initializers are matched — bitwise,
transposed — against the base checkpoint before training starts, so the mapping is
proven, not guessed. The acoustic graph is re-exported from a functional mirror of
``AcousticDecoder.cached_step`` (identical IO: token_emb/position_ids/past_k_0/
past_v_0 -> hidden/present_k_0/present_v_0), because ``nn.RMSNorm``'s fused
``aten::rms_norm`` has no ONNX op.

    uv run python finetune/qat_v3turbo.py \
        --data finetune/dataset/dograh_tts_dataset/train_2500.parquet --run qat1
    # sanity: PTQ baseline through this very pipeline (no training, base weights)
    uv run python finetune/qat_v3turbo.py --export-only --run ptq0
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from vieneu_lora import V3TurboLoraDataset, compute_loss, load_rows                        # noqa: E402
from vieneu_lora.utils import safe_path                                                    # noqa: E402

# ── deployment quantization arithmetic (ORT quantize_dynamic, reduce_range=True) ──
QMAX = 64  # 7-bit symmetric weights: scale = absmax / 64


def fq_weight(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric 7-bit fake-quant of a Linear weight (out, in)."""
    s = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / QMAX
    q = torch.clamp(torch.round(w.detach() / s), -QMAX, QMAX)
    out = q * s
    return w + (out - w).detach()  # STE: forward quantized, backward identity


def fq_act(a: torch.Tensor) -> torch.Tensor:
    """Per-tensor dynamic asymmetric uint8 fake-quant (ORT DynamicQuantizeLinear)."""
    a = a.float()
    amin = a.detach().amin(); amax = a.detach().amax()
    s = ((amax - amin) / 255.0).clamp(min=1e-12)
    zp = torch.clamp(torch.round(-amin / s), 0.0, 255.0)
    q = torch.clamp(torch.round(a / s) + zp, 0.0, 255.0)
    out = (q - zp) * s
    return a + (out - a).detach()


# ── deployment quantization arithmetic (ORT MatMulNBits, bits=4|2) ────────────
# Mirrors ORT MLAS `BlockwiseQuantizer` (core/mlas/lib/q4_dq.cpp — the same C++
# template backs onnxruntime.capi.quantize_matmul_2bits / _4bits / _8bits) in
# asymmetric mode (the DefaultWeightOnlyQuantConfig default), per (out-channel,
# block of `block_size` along the input dim), with levels = 2**bits − 1:
#   mn = min(block, 0); mx = max(block, 0)   # range always includes 0
#   s  = (mx - mn) / levels                  # fp32 (15 for int4, 3 for int2)
#   zp = clamp(roundf(-mn / s), 0, levels)   # DIVISION by scale; roundf = ties away
#   q  = clamp(roundf(w * (1/s) + zp), 0, levels)  # RECIPROCAL multiply — NOT w/s
#   deq = (q - zp) * s
# The three bold-face details (0 in range, zp via division, q via reciprocal)
# are invisible on random data but change ~0.2% of codes on real bf16-derived
# weights, whose quotients land on exact .5 ties. Weight-only: activations fp32.
# int2 pinned bitwise the same way (levels=3, verified on a live MatMulNBits
# bits=2 CPU session: max|Δ| 6e-07).
def _roundf(x: torch.Tensor) -> torch.Tensor:
    """C roundf (nearest, ties away from zero) — exact for fp32 inputs.

    x + 0.5 is exact in float64 for fp32 x, so the double hop matches roundf
    even on ties and near-ties where floor(x + 0.5) in fp32 would misround.
    """
    xd = x.double()
    return torch.where(xd >= 0, torch.floor(xd + 0.5), -torch.floor(-xd + 0.5)).float()


def _quant_nbits(wb: torch.Tensor, levels: float):
    """wb: (out, n_blocks, bs) → dequantized grid values (same shape)."""
    z = torch.zeros((), dtype=wb.dtype, device=wb.device)
    mn = torch.minimum(wb.amin(-1), z)                  # range always includes 0
    mx = torch.maximum(wb.amax(-1), z)
    s = (mx - mn) / levels
    ok = s != 0                                         # all-zero block: s = 0
    s_safe = torch.where(ok, s, torch.ones_like(s))
    zp = torch.clamp(_roundf(torch.where(ok, -mn / s_safe, mn)), 0.0, levels)
    recip = torch.where(ok, 1.0 / s_safe, torch.zeros_like(s))
    q = torch.clamp(_roundf(wb * recip[..., None] + zp[..., None]), 0.0, levels)
    return (q - zp[..., None]) * s[..., None]


def fq_weight_nbits(w: torch.Tensor, bs: int, levels: float) -> torch.Tensor:
    """Blockwise asymmetric uintN fake-quant along the input dim (ORT MatMulNBits)."""
    out_dim, in_dim = w.shape
    if in_dim % bs:
        raise ValueError(f"in_features {in_dim} not divisible by block_size {bs}")
    wq = _quant_nbits(w.detach().reshape(out_dim, in_dim // bs, bs), levels).reshape_as(w)
    return w + (wq - w).detach()   # STE: forward quantized, backward identity


class QLinear(nn.Linear):
    """nn.Linear with the deployment fake-quant in front (state_dict-compatible).

    Swapping ``module.__class__`` keeps parameters, buffers and state_dict keys
    untouched — only the forward changes.
    """

    enabled = False        # class-level switch: eval fp32 (False) vs QAT (True)
    quant_act = True       # int8 mode only
    bits = 8               # 8 (dynamic per-channel 7-bit + act uint8) | 4 | 2 (MatMulNBits weight-only)
    block_size = 32        # int4/int2 mode: block along the input dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not QLinear.enabled:
            return super().forward(x)
        if QLinear.bits in (4, 2):
            w = fq_weight_nbits(self.weight, QLinear.block_size, 2.0 ** QLinear.bits - 1)  # act stays fp32
        else:
            if QLinear.quant_act:
                x = fq_act(x)
            w = fq_weight(self.weight)
        return F.linear(x, w, self.bias)


def attach_qat(model, quant_act: bool = True, bits: int = 8, block_size: int = 32) -> int:
    """Swap every nn.Linear inside backbone + acoustic decoder for QLinear."""
    QLinear.quant_act = quant_act
    QLinear.bits = bits
    QLinear.block_size = block_size
    n = 0
    for root_name in ("semantic_backbone", "acoustic_decoder"):
        for mod in getattr(model, root_name).modules():
            if type(mod) is nn.Linear:
                mod.__class__ = QLinear
                mod.weight.requires_grad_(True)
                n += 1
    return n


def bake(model) -> None:
    """Freeze the fake-quant: weight.data := quantized grid.

    int8: exact under re-quantization. int4/int2: re-quantizing the baked grid
    moves a small fraction of codes by one level (int4: ~0.07%, max 0.5 grid
    step, measured on the pretrained weights) — the fake-quant itself is
    bitwise-identical to the kernel."""
    QLinear.enabled = False
    for mod in model.modules():
        if isinstance(mod, QLinear):
            with torch.no_grad():
                if QLinear.bits in (4, 2):
                    w = mod.weight
                    mod.weight.copy_(fq_weight_nbits(w, QLinear.block_size, 2.0 ** QLinear.bits - 1).detach())
                else:
                    mod.weight.copy_(_grid(mod.weight))


def _grid(w: torch.Tensor) -> torch.Tensor:
    s = w.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / QMAX
    return torch.clamp(torch.round(w.detach() / s), -QMAX, QMAX) * s


# ── ONNX weight transplant ─────────────────────────────────────────────────────
def discover_mapping(model, ref_dir: Path) -> dict:
    """Bitwise-match the anonymous MatMul initializers to torch backbone weights."""
    import onnx
    from safetensors.torch import load_file

    snap = model._qat_snapshot  # set in main(): path of the local checkpoint dir
    sd = load_file(str(Path(snap) / "model.safetensors"))
    cand = {k: v for k, v in sd.items() if k.startswith("semantic_backbone.") and v.ndim == 2}
    mapping = {}
    for f in ("vieneu_prefill.onnx", "vieneu_decode_step.onnx"):  # names differ per graph!
        m = onnx.load(str(ref_dir / f))  # loads external .data
        arrs = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}
        fm = {}
        for name, a in arrs.items():
            if not name.startswith("onnx::MatMul"):
                continue
            hits = [k for k, v in cand.items() if v.T.shape == a.shape and np.array_equal(v.float().T.numpy(), a)]
            if len(hits) != 1:
                raise RuntimeError(f"weight mapping failed for {f}/{name}: {len(hits)} bitwise hits")
            fm[name] = hits[0]
        if len(fm) != 84:
            raise RuntimeError(f"{f}: expected 84 backbone MatMul weights, matched {len(fm)}")
        mapping[f] = fm
    return mapping


def transplant_backbone(model, mapping: dict, ref_dir: Path, out_dir: Path) -> None:
    """Re-emit the fp32 reference graphs with the (baked) torch weights inside."""
    import onnx

    out_dir.mkdir(parents=True, exist_ok=True)
    # onnx.save_model APPENDS to an existing external-data file — clear stale output
    for stale in out_dir.iterdir():
        stale.unlink()
    live = dict(model.state_dict())
    for f in ("vieneu_prefill.onnx", "vieneu_decode_step.onnx"):
        new = {name: live[key].detach().cpu().T.contiguous() for name, key in mapping[f].items()}
        m = onnx.load(str(ref_dir / f))
        inits = m.graph.initializer
        replaced = 0
        for i, t in enumerate(inits):
            if t.name in new:
                inits[i].CopyFrom(onnx.numpy_helper.from_array(new[t.name].numpy(), t.name))
                replaced += 1
        if replaced != len(new):
            raise RuntimeError(f"{f}: replaced {replaced}/{len(new)} weights")
        # NOTE onnx.save_model APPENDS to the shared .data on each save, so the
        # second graph's weights land as a second copy (~+415 MB); _dedupe_shared_data
        # truncates it back when both graphs emit identical offset maps
        onnx.save_model(m, str(out_dir / f), save_as_external_data=True,
                        location="vieneu_backbone_shared.data", size_threshold=1024)
    _dedupe_shared_data(out_dir)
    for aux in ("config.json", "tokenizer.json", "vieneu_v3_heads.npz"):
        src = ref_dir / aux
        if src.is_file():
            (out_dir / aux).write_bytes(src.read_bytes())


def _dedupe_shared_data(out_dir: Path) -> None:
    """onnx.save_model APPENDS each save's tensors to the same .data file; when both
    graphs emit identical (offset, length) maps — same initializers, same order —
    the second appended copy is redundant and the file can be truncated."""
    import onnx
    from onnx.external_data_helper import uses_external_data

    maps = []
    for f in ("vieneu_prefill.onnx", "vieneu_decode_step.onnx"):
        m = onnx.load(str(out_dir / f), load_external_data=False)
        fm = {}
        for t in m.graph.initializer:
            if uses_external_data(t):
                d = {r.key: r.value for r in t.external_data}
                fm[t.name] = (int(d["offset"]), int(d["length"]))
        maps.append(fm)
    if maps[0] and maps[0] == maps[1]:
        end = max(o + l for o, l in maps[0].values())
        (out_dir / "vieneu_backbone_shared.data").open("r+b").truncate(end)


# ── acoustic graph export (functional mirror of cached_step) ───────────────────
def export_acoustic(model, out_path: Path) -> None:
    dec = model.acoustic_decoder.float().cpu()   # trace on CPU
    layer, a = dec.layers[0], dec.layers[0].attn
    pos, norm_w, eps = dec.slot_pos_emb, dec.norm.weight, float(dec.norm.eps)

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    class Step(nn.Module):
        def forward(self, token_emb, position_ids, past_k_0, past_v_0):
            x = token_emb + pos(position_ids)
            S, P = x.shape[1], past_k_0.shape[2]
            h = rms(x, layer.norm1.weight)
            qkv = F.linear(h, a.qkv.weight).reshape(1, S, 3, a.num_heads, a.head_dim)
            q, k, v = qkv.unbind(2)
            q = rms(q.transpose(1, 2), a.q_norm.weight)
            k = rms(k.transpose(1, 2), a.k_norm.weight)
            v = v.transpose(1, 2)
            k = torch.cat([past_k_0, k], 2); v = torch.cat([past_v_0, v], 2)
            P1 = k.shape[2]
            idx_k = torch.arange(P1, dtype=position_ids.dtype)
            causal = (idx_k.unsqueeze(0) <= (torch.arange(S) + P).unsqueeze(1)).to(x.dtype)
            bias = torch.where(causal > 0, torch.zeros_like(causal), torch.full_like(causal, -1e9))
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(a.head_dim) + bias, -1)
            hd = a.qkv.weight.shape[0] // 3
            x = x + F.linear((att @ v).transpose(1, 2).reshape(1, S, hd), a.o_proj.weight)
            n2 = rms(x, layer.norm2.weight)
            x = x + F.linear(F.silu(F.linear(n2, layer.ff_gate.weight)) * F.linear(n2, layer.ff_up.weight),
                             layer.ff_down.weight)
            return rms(x, norm_w), k, v

    m = Step().eval()
    for q in dec.parameters():
        q.requires_grad_(False)
    tok = torch.randn(1, 2, 768); p = torch.tensor([[0, 1]])
    pk = torch.randn(1, 8, 5, 96); pv = torch.randn(1, 8, 5, 96)
    kw = dict(opset_version=17, input_names=["token_emb", "position_ids", "past_k_0", "past_v_0"],
              output_names=["hidden", "present_k_0", "present_v_0"],
              dynamic_axes={"token_emb": {1: "S"}, "position_ids": {1: "S"}, "past_k_0": {2: "P"},
                            "past_v_0": {2: "P"}, "hidden": {1: "S"}, "present_k_0": {2: "P1"},
                            "present_v_0": {2: "P1"}}, do_constant_folding=True)
    with torch.no_grad():
        try:
            torch.onnx.export(m, (tok, p, pk, pv), str(out_path), dynamo=False, **kw)
        except TypeError:  # older torch without the dynamo kwarg
            torch.onnx.export(m, (tok, p, pk, pv), str(out_path), **kw)
    _promote_constants(out_path)


def _promote_constants(path: Path) -> None:
    """Turn Constant nodes into initializers so quantize_dynamic can see the weights.

    Traced with requires_grad=False the exporter folds Linear weights into
    Constant nodes; quantize_dynamic only quantizes MatMuls whose B input is an
    initializer, so the acoustic graph would silently stay fp32.
    """
    import onnx

    m = onnx.load(str(path))
    g = m.graph
    keep = []
    for n in g.node:
        if n.op_type == "Constant" and len(n.attribute) == 1 and n.attribute[0].name == "value":
            t = onnx.helper.get_attribute_value(n.attribute[0])
            t = t if isinstance(t, onnx.TensorProto) else onnx.numpy_helper.from_array(
                onnx.numpy_helper.to_array(t), n.output[0])
            t.name = n.output[0]
            g.initializer.append(t)
        else:
            keep.append(n)
    del g.node[:]
    g.node.extend(keep)
    onnx.save(m, str(path))


def quantize_dir(fp32_dir: Path, q_dir: Path, bits: int = 8, block_size: int = 32) -> None:
    """int8: shipped recipe (pinned bitwise against onnx_int8).
    int4/int2: ORT MatMulNBits weight-only blockwise (arithmetic pinned in fq_weight_nbits)."""
    q_dir.mkdir(parents=True, exist_ok=True)
    graphs = ("vieneu_prefill.onnx", "vieneu_decode_step.onnx", "vieneu_acoustic_cached.onnx")
    if bits in (4, 2):
        import logging
        logging.disable(logging.INFO)   # quantizer logs every node
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig)
        for f in graphs:
            q = MatMulNBitsQuantizer(str(fp32_dir / f),
                                     algo_config=DefaultWeightOnlyQuantConfig(bits=bits, block_size=block_size))
            q.process()
            q.model.save_model_to_file(str(q_dir / f), use_external_data_format=False)
            # onnx-ir may write IR 14; ORT 1.31 runtime only loads ≤ 13
            m = onnx.load(str(q_dir / f))
            if m.ir_version > 13:
                m.ir_version = 9
                onnx.save(m, str(q_dir / f))
    else:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        for f in graphs:
            quantize_dynamic(str(fp32_dir / f), str(q_dir / f), op_types_to_quantize=["MatMul"],
                             per_channel=True, weight_type=QuantType.QInt8, reduce_range=True)
    for aux in ("config.json", "tokenizer.json", "vieneu_v3_heads.npz"):
        src = fp32_dir / aux
        if src.is_file():
            (q_dir / aux).write_bytes(src.read_bytes())


# ── verification ───────────────────────────────────────────────────────────────
@torch.no_grad()
def self_check(model, fp32_dir: Path, q_dir: Path, bits: int = 8) -> None:
    import onnxruntime as ort

    so = ort.SessionOptions()
    prov = ["CPUExecutionProvider"]
    rng = np.random.default_rng(0)
    model = model.float().cpu().eval()
    emb = torch.from_numpy(rng.standard_normal((1, 17, 768)).astype(np.float32))
    mask = torch.ones((1, 17), dtype=torch.bool)
    spk = torch.from_numpy(rng.standard_normal((1, 192)).astype(np.float32))
    QLinear.enabled = False
    with torch.no_grad():
        h_t = model.semantic_backbone(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                      return_dict=True).last_hidden_state
    s = ort.InferenceSession(str(fp32_dir / "vieneu_prefill.onnx"), so, providers=prov)
    h_o = s.run(None, {"inputs_embeds": emb.numpy()})[0]
    print(f"  [check] fp32 prefill vs torch: max|Δ| = {np.abs(h_t.numpy() - h_o).max():.2e}")
    if bits in (4, 2):
        # torch fake-quant (deployment arithmetic) vs the MatMulNBits graph
        QLinear.enabled = True
        with torch.no_grad():
            h_q = model.semantic_backbone(inputs_embeds=emb, attention_mask=mask, use_cache=False,
                                          return_dict=True).last_hidden_state
        QLinear.enabled = False
        sq = ort.InferenceSession(str(q_dir / "vieneu_prefill.onnx"), so, providers=prov)
        h_qo = sq.run(None, {"inputs_embeds": emb.numpy()})[0]
        print(f"  [check] int{bits} prefill vs torch fake-quant: max|Δ| = {np.abs(h_q.numpy() - h_qo).max():.2e}")
    for f in ("vieneu_prefill.onnx", "vieneu_decode_step.onnx", "vieneu_acoustic_cached.onnx"):
        ort.InferenceSession(str(q_dir / f), so, providers=prov)  # must load + init
    print(f"  [check] int{bits} sessions load OK")


# ── training ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_loss(model, loader, device, autocast, args):
    model.eval()
    tot, acc, n = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            _, m = compute_loss(model, batch, args.text_loss_weight, args.audio_loss_weight)
        tot += m["audio_loss"]; acc += m["acc_cb0"]; n += 1
    model.train()
    return tot / max(n, 1), acc / max(n, 1)


def train(args, model, train_loader, eval_loader, device, autocast) -> None:
    from transformers import get_scheduler

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.98), eps=1e-8)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    sched = get_scheduler("cosine", opt, int(total * args.warmup_ratio), total)
    print(f"QAT training: {total} optimizer steps ({steps_per_epoch}/epoch), "
          f"lr {args.lr}, act-quant={QLinear.quant_act}, fake-quant={not args.no_fake_quant}")

    # evidence table: fp32 vs PTQ (fake-quant, untrained) on the eval split
    QLinear.enabled = False
    l_fp32, acc = eval_loss(model, eval_loader, device, autocast, args)
    QLinear.enabled = not args.no_fake_quant
    l_ptq, _ = eval_loss(model, eval_loader, device, autocast, args)
    print(f"  eval audio CE — fp32 {l_fp32:.4f} | PTQ(fake-quant) {l_ptq:.4f} "
          f"| gap {l_ptq - l_fp32:+.4f} | acc_cb0(fp32) {acc:.3f}   ← QAT must close the gap")

    model.train()
    t0 = time.time(); running = 0.0
    step = 0
    while step < total:
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                loss, m = compute_loss(model, batch, args.text_loss_weight, args.audio_loss_weight)
            (loss / args.grad_accum).backward()
            running += loss.item()
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    l_ev, acc_ev = eval_loss(model, eval_loader, device, autocast, args)
                    print(f"  step {step}/{total}  train {running / args.log_every / args.grad_accum:.4f}  "
                          f"eval {l_ev:.4f} (fp32 {l_fp32:.4f}, ptq {l_ptq:.4f})  acc_cb0 {acc_ev:.3f}  "
                          f"{(time.time() - t0) / step:.2f}s/step")
                    running = 0.0
                if step >= total:
                    break
    l_ev, acc_ev = eval_loss(model, eval_loader, device, autocast, args)
    print(f"final eval audio CE: {l_ev:.4f}  (fp32 {l_fp32:.4f}, untrained PTQ {l_ptq:.4f})  "
          f"acc_cb0 {acc_ev:.3f}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "finetune/dataset/dograh_tts_dataset/train_2500.parquet"))
    ap.add_argument("--run", default="qat1")
    ap.add_argument("--output-dir", default=str(ROOT / "finetune/output/qat"))
    ap.add_argument("--snapshot", default="", help="local dir with update/model.safetensors (default: HF cache)")
    ap.add_argument("--reference-onnx", default=str(ROOT / "finetune/output/qat/reference/onnx_update"),
                    help="shipped fp32 graphs whose weights are transplanted")
    # quantization
    ap.add_argument("--no-act-quant", action="store_true", help="fake-quant weights only")
    ap.add_argument("--bits", type=int, choices=[8, 4, 2], default=8,
                    help="8 = shipped dynamic int8 recipe | 4 | 2 = ORT MatMulNBits weight-only blockwise")
    ap.add_argument("--block-size", type=int, default=32, help="int4/int2 block along the input dim (32/64/128)")
    ap.add_argument("--no-fake-quant", action="store_true",
                    help="control experiment: plain fp32 training (identity forward), no bake — "
                         "export becomes PTQ of the finetuned weights")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--eval-ratio", type=float, default=0.03)
    ap.add_argument("--text-loss-weight", type=float, default=1.0)
    ap.add_argument("--audio-loss-weight", type=float, default=8.0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast during QAT")
    ap.add_argument("--export-only", action="store_true", help="skip training: bake base weights and export (PTQ baseline)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise SystemExit("QAT training needs a CUDA GPU (use --export-only for the CPU path).")
    out_dir = safe_path(Path(args.output_dir) / args.run)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")

    snapshot = args.snapshot or glob.glob(str(Path.home() /
        ".cache/huggingface/hub/models--pnnbao-ump--VieNeu-TTS-v3-Turbo/snapshots/*"))[0] + "/update"

    from transformers import AutoTokenizer
    from vieneu._v3_turbo_engine.hub_load_v3_turbo import load_v3_turbo_checkpoint
    print(f"loading {snapshot} ...")
    model = load_v3_turbo_checkpoint(snapshot, device=device, dtype=torch.float32)
    model.config.use_cache = False
    model._qat_snapshot = snapshot
    for p in model.parameters():
        p.requires_grad_(False)   # only the QAT Linears' weights train (see attach_qat)

    n_lin = attach_qat(model, quant_act=(not args.no_act_quant and args.bits == 8),
                       bits=args.bits, block_size=args.block_size)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"QAT Linears: {n_lin} (84 backbone + 5 acoustic)  trainable {n_par/1e6:.1f}M params")

    mapping = discover_mapping(model, Path(args.reference_onnx))
    (out_dir / "mapping.json").write_text(json.dumps(mapping, indent=1))
    print(f"weight mapping: {sum(len(v) for v in mapping.values())} MatMul inits matched bitwise (84 per graph)")

    rows = load_rows(safe_path(args.data))
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * args.eval_ratio))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    train_ds = V3TurboLoraDataset(train_rows, tokenizer, model.config, max_length=args.max_length)
    eval_ds = V3TurboLoraDataset(eval_rows, tokenizer, model.config, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, collate_fn=train_ds.collate, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=eval_ds.collate)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_bf16)
    if not args.export_only:
        train(args, model, train_loader, eval_loader, device, autocast)

    if args.no_fake_quant:
        print("control run: skipping bake — plain fp32 weights, export = PTQ of finetuned weights")
        QLinear.enabled = False
    else:
        print("baking fake-quant into weights ...")
        bake(model)

    fp32_dir = out_dir / "onnx_qat_fp32"
    q_dir = out_dir / f"onnx_qat_int{args.bits}"
    print("exporting backbone (weight transplant) ...")
    transplant_backbone(model, mapping, Path(args.reference_onnx), fp32_dir)
    print("exporting acoustic graph ...")
    export_acoustic(model, fp32_dir / "vieneu_acoustic_cached.onnx")
    print(f"quantizing (bits={args.bits}"
          + (f", block_size={args.block_size}" if args.bits != 8 else " — pinned shipped recipe") + ") ...")
    quantize_dir(fp32_dir, q_dir, bits=args.bits, block_size=args.block_size)

    from safetensors.torch import save_model
    save_model(model, str(out_dir / "qat_model.safetensors"))
    self_check(model, fp32_dir, q_dir, bits=args.bits)
    print(f"done → {q_dir}  (engine: Vieneu(mode='v3turbo', precision='int8', onnx_dir=...))")


if __name__ == "__main__":
    main()
