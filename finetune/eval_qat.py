"""Evaluate QAT int8 graphs: greedy code agreement + CPU speed.

Drives the three ONNX sessions of each graph directory exactly the way
``OnnxV3LiteEngine`` does (numpy heads from vieneu_v3_heads.npz, prefill →
acoustic chain → decode_step), but with greedy sampling so the comparison is
deterministic. Reports, per engine and against the baseline engine:

    - frame-level audio-code agreement (mean over 16 codebooks, on the common
      prefix of frames) — the quantization fidelity metric
    - EOS agreement and generated-frame count
    - RTF of the TTS model itself (prefill + decode_step + acoustic chain,
      without the audio codec) on this CPU

    .venv/bin/python finetune/eval_qat.py \
        fp32=finetune/output/qat/reference/onnx_update \
        ptq=finetune/output/qat/reference/onnx_int8 \
        qat=finetune/output/qat/qat1/onnx_qat_int8 \
        --data finetune/dataset/dograh_tts_dataset/train_2500.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "finetune"))

from vieneu_lora import load_rows                                                    # noqa: E402
from vieneu_lora.utils import safe_path                                              # noqa: E402


class Graphs:
    """One graph dir: sessions + numpy heads, mirroring OnnxV3LiteEngine."""

    def __init__(self, path: Path, threads: int = 0):
        import onnxruntime as ort

        c = json.loads((path / "config.json").read_text(encoding="utf-8"))
        self.name = path.parent.name + "/" + path.name
        self.n_vq, self.H = int(c["n_vq"]), int(c["hidden_size"])
        self.L = int(c["num_hidden_layers"])
        self.L_loc = int(c.get("local_num_hidden_layers", 1))
        self.audio_pad = int(c["audio_pad_token_id"])
        self.tps, self.tpe = int(c["text_prompt_start_token_id"]), int(c["text_prompt_end_token_id"])
        self.sgs = int(c["speech_generation_start_token_id"])
        self.eos_speech = int(c["speech_generation_end_token_id"])
        self.default_style_id = int(c.get("default_style_token_id", 16))
        z = np.load(path / "vieneu_v3_heads.npz")
        self.text_emb = z["text_emb"].astype(np.float32)
        self.audio_emb = z["audio_emb"].astype(np.float32)
        self.xvec_w = z["xvec_w"].astype(np.float32) if "xvec_w" in z else None
        self.xvec_b = z["xvec_b"].astype(np.float32) if "xvec_b" in z else None
        self.xvec_ln_w = z["xvec_ln_w"].astype(np.float32) if "xvec_ln_w" in z else None
        self.xvec_ln_b = z["xvec_ln_b"].astype(np.float32) if "xvec_ln_b" in z else None
        self.xvec_ln_eps = float(z["xvec_ln_eps"]) if "xvec_ln_eps" in z else 1e-5
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = threads or max(1, min(int(__import__("os").cpu_count() or 8), 16))
        prov = ["CPUExecutionProvider"]
        self.pre = ort.InferenceSession(str(path / "vieneu_prefill.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(path / "vieneu_decode_step.onnx"), so, providers=prov)
        self.ac = ort.InferenceSession(str(path / "vieneu_acoustic_cached.onnx"), so, providers=prov)

    def anchor(self, speaker_emb: np.ndarray) -> np.ndarray:
        v = np.asarray(speaker_emb, dtype=np.float32).reshape(-1)
        v = v @ self.xvec_w.T + self.xvec_b
        v = (v - v.mean()) / np.sqrt(v.var() + self.xvec_ln_eps)
        return (v * self.xvec_ln_w + self.xvec_ln_b).astype(np.float32)

    def rows(self, phones: str) -> np.ndarray:
        ids = self.tokenizer.encode(phones, add_special_tokens=False).ids
        text_ids = [self.default_style_id, self.tps, *ids, self.tpe]
        rows = np.full((len(text_ids), self.n_vq + 1), self.audio_pad, dtype=np.int64)
        rows[:, 0] = text_ids
        return rows

    def embed(self, rows: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        emb = self.text_emb[rows[:, 0]].copy()
        for ch in range(self.n_vq):
            ids = rows[:, ch + 1]
            valid = ids != self.audio_pad
            emb += self.audio_emb[ch][np.where(valid, ids, 0)] * valid[:, None]
        return (emb + anchor[None])[None].astype(np.float32)

    def gen(self, phones: str, speaker_emb: np.ndarray, max_frames: int) -> tuple[np.ndarray, bool]:
        """Greedy generation → (codes (T, n_vq), hit_eos)."""
        anchor = self.anchor(speaker_emb)
        prompt = self.embed(self.rows(phones), anchor)
        T0 = prompt.shape[1]
        t_pre = time.perf_counter()
        pre = self.pre.run(None, {"inputs_embeds": prompt})
        past_k = [pre[1 + i] for i in range(self.L)]
        past_v = [pre[1 + self.L + i] for i in range(self.L)]
        h = pre[0][:, -1]
        frames, hit_eos = [], False
        empty = np.zeros((1, 8, 0, self.H // 8), dtype=np.float32)
        for t in range(max_frames):
            # acoustic chain: [cond, text-token] → cb0, then cb1..cb15 one by one
            txt = self.text_emb[self.sgs].astype(np.float32)
            feed = {"token_emb": np.stack([h[0].astype(np.float32), txt])[None],
                    "position_ids": np.array([[0, 1]], np.int64),
                    "past_k_0": empty, "past_v_0": empty}
            out = self.ac.run(None, feed)
            hidden, pk, pv = out[0], out[1:1 + self.L_loc], out[1 + self.L_loc:1 + 2 * self.L_loc]
            logits = hidden[0, 0].astype(np.float32) @ self.text_emb.T
            eos = int(logits.argmax()) == self.eos_speech
            codes = [int((hidden[0, 1].astype(np.float32) @ self.audio_emb[0].T).argmax())]
            for ch in range(1, self.n_vq):
                emb = self.audio_emb[ch - 1][codes[-1]].astype(np.float32)
                feed = {"token_emb": emb.reshape(1, 1, self.H),
                        "position_ids": np.array([[ch + 1]], np.int64)}
                for i in range(self.L_loc):
                    feed[f"past_k_{i}"] = pk[i]
                    feed[f"past_v_{i}"] = pv[i]
                out = self.ac.run(None, feed)
                hidden = out[0]
                pk, pv = out[1:1 + self.L_loc], out[1 + self.L_loc:1 + 2 * self.L_loc]
                codes.append(int((hidden[0, 0].astype(np.float32) @ self.audio_emb[ch].T).argmax()))
            frames.append(codes)
            if eos:
                hit_eos = True
                break
            slot = np.full((1, 1, self.n_vq + 1), self.audio_pad, dtype=np.int64)
            slot[:, :, 0] = self.sgs
            slot[0, 0, 1:] = codes
            se = self.embed(slot[0], anchor)
            feed = {"inputs_embeds": se, "position_ids": np.array([[T0 + t]], np.int64)}
            for i in range(self.L):
                feed[f"past_k_{i}"] = past_k[i]
                feed[f"past_v_{i}"] = past_v[i]
            out = self.dec.run(None, feed)
            h = out[0][:, 0]
            past_k = [out[1 + i] for i in range(self.L)]
            past_v = [out[1 + self.L + i] for i in range(self.L)]
        self.last_time = time.perf_counter() - t_pre
        return np.asarray(frames, dtype=np.int64), hit_eos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("engines", nargs="+", help="name=graph_dir pairs; the FIRST is the agreement baseline")
    ap.add_argument("--data", default=str(ROOT / "finetune/dataset/dograh_tts_dataset/train_2500.parquet"))
    ap.add_argument("--n-texts", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=180)
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = all cores)")
    ap.add_argument("--wav-dir", default="", help="also write one 48 kHz wav per engine/text (needs the MOSS codec in the HF cache)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    engines = {}
    for spec in args.engines:
        name, path = spec.split("=", 1)
        engines[name] = Graphs(Path(path), args.threads)
    rows = load_rows(safe_path(args.data))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=min(args.n_texts, len(rows)), replace=False)
    texts = [rows[i]["phones"] for i in idx]
    speaker = np.mean([np.asarray(r["speaker_embedding"], np.float32) for r in rows], axis=0)

    results: dict[str, list] = {name: [] for name in engines}
    eos_flags: dict[str, list] = {name: [] for name in engines}
    times: dict[str, list] = {name: [] for name in engines}
    codec = None
    if args.wav_dir:
        import onnxruntime as ort

        cd = next(Path.home().glob(".cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano-ONNX/snapshots/*"))
        so = ort.SessionOptions()
        codec = ort.InferenceSession(str(cd / "moss_audio_tokenizer_decode_full.onnx"), so,
                                     providers=["CPUExecutionProvider"])
        Path(args.wav_dir).mkdir(parents=True, exist_ok=True)
    base_name = args.engines[0].split("=", 1)[0]
    for k, phones in enumerate(texts):
        base_codes = None
        for name, g in engines.items():
            codes, eos = g.gen(phones, speaker, args.max_frames)
            results[name].append(codes)
            eos_flags[name].append(eos)
            times[name].append(g.last_time)
            if codec is not None and codes.shape[0]:
                c = codes.astype(np.int32)[None]
                wav = codec.run(None, {"audio_codes": c,
                                       "audio_code_lengths": np.array([c.shape[1]], np.int32)})[0][0].mean(0)
                pcm = (np.clip(wav, -1, 1) * 32767).astype("<i2")
                import wave
                with wave.open(str(Path(args.wav_dir) / f"{name}_{k:02d}.wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000)
                    w.writeframes(pcm.tobytes())
            if name == base_name:
                base_codes = codes
        # agreement on the common prefix of frames
        T = min(results[name][k].shape[0] for name in engines)
        line = []
        for name in engines:
            agree = float(np.mean(results[name][k][:T] == base_codes[:T]))
            line.append(f"{name} {agree*100:5.1f}%")
        print(f"[{k+1}/{len(texts)}] frames " + " | ".join(line))

    print(f"\n=== greedy agreement vs baseline '{base_name}' (mean over {len(texts)} texts) ===")
    for name in engines:
        ags = []
        for k in range(len(texts)):
            T = min(results[name][k].shape[0], results[base_name][k].shape[0])
            ags.append(float(np.mean(results[name][k][:T] == results[base_name][k][:T])))
        t = np.sum(times[name])
        frames = sum(r.shape[0] for r in results[name])
        rtf = t / (frames / 12.5)
        eos_agree = float(np.mean([a == b for a, b in zip(eos_flags[name], eos_flags[base_name])]))
        print(f"  {name:28s} agree {np.mean(ags)*100:5.1f}%  eos {eos_agree*100:5.1f}%  "
              f"frames/s {frames/t:6.1f}  RTF {rtf:5.2f}x")


if __name__ == "__main__":
    main()
