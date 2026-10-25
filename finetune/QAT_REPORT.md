# Báo cáo: Quantization-Aware Training (QAT) int8, int4 & int2 cho VieNeu v3 Turbo

**Ngày:** 2026-10-23 · **Máy:** RTX PRO 6000 96 GB + Xeon 16-core (avx512_vnni) · **Môi trường:** `.venv` của repo (torch 2.13.0+cu130, transformers 4.57.6, onnx 1.23.2, onnxruntime 1.31.0)

---

## 1. Tóm tắt (TL;DR)

Fine-tune model v3 Turbo theo phương pháp **QAT** trên dataset dograh (2.389 mẫu) sao cho weights "sống hòa" với arithmetic int8 của bản deploy, thay vì bị lượng tử một chiều (PTQ) như bản shipped.

| Kích thước | Bản fp32 gốc | **Bản int8 của họ (PTQ)** | **QAT int8 (báo cáo này)** |
|---|---|---|---|
| Dung lượng | 454 MB | 158 MB | **158 MB** (bằng đúng) |
| Tốc độ model (≈1 s audio, CPU 16 luồng) | 176 ms | 123 ms (1.43×) | **100 ms (1.76×)** |
| Sát thương CE do quantize (teacher-forced) | — | +0.1032 | **+0.0556 (−46%)** |
| Nhiễu hidden state decode_step | — | relFro 3.25% | **1.85% (−43%)** |
| Top-1 agreement vs fp32 twin | — | 59.3% | **64.4%** |

**Kết luận:** giữ nguyên recipe/size/tốc độ của bản int8 shipped, chất lượng sau quantize tốt hơn rõ rệt. **Control experiment** (mục 6.1): finetune fp32 thuần cùng data/steps rồi PTQ vẫn chịu sát thương quantize đầy đủ (+0.114) → mức giảm của QAT là công dụng thật của QAT, không phải tác dụng phụ của finetune. Deploy được ngay:

```python
VieNeu(mode="v3turbo", precision="int8", onnx_dir="finetune/output/qat/qat1/onnx_qat_int8")
```

**Bổ sung int4 (mục 10):** QAT int4 weight-only (`MatMulNBits`, block 32) **thắng QAT int8 trên mọi metric chất lượng** — chi phí quantize CE **+0.0000** (vs +0.056), hidden perturbation **0.01%** (vs 8%), greedy agreement **93.1%** (vs 37.3%) — nhỏ hơn 21% (**124 MB** vs 158 MB), decode nhanh hơn 10%, end-to-end production **ngang int8** (RTF 0.34× cả hai). Cảnh báo: **PTQ int4 không dùng được** (free-running hỏng: agreement 17%, EOS 58%) — int4 bắt buộc phải QAT.

**Bổ sung int2 (mục 11):** 2-bit **chạy được về chất lượng nhưng vô lý về tốc độ trên CPU**. QAT int2 (levels=3, cùng recipe) đạt hidden relFro **0.03%**, agreement **69.5%**, EOS 83.3%, quant cost CE +0.0000 — trần CE thấp hơn int4 một chút (4.457 vs 4.175). Nhưng kernel CPU bits=2 của ORT **không có assembly MLAS** (chỉ Q4/Q8Q4 có) → decode_step **71 ms/frame = chậm 14× int4**, production RTF 2.0–2.3× (chậm hơn realtime). Size 92 MB. **Kết luận: trên CPU, int4 vẫn là điểm ngọt; int2 chỉ đáng thử trên runtime có kernel 2-bit tối ưu (GPU/TRT).**

---

## 2. Bối cảnh & mục tiêu

- Model chạy production qua `src/vieneu/_v3_turbo_engine/onnx_runtime_lite.py`: **3 graph ONNX + 1 file heads numpy**, chạy CPU-ONNX.
- Bản shipped có sẵn 2 mức: `onnx_update` (gọi là "fp32") và `onnx_int8`. Không có tài liệu nào mô tả cách tạo bản int8.
- Mục tiêu: (a) tái hiện chính xác recipe int8 của họ; (b) fine-tune để giảm mất mát khi quantize; (c) so sánh chi tiết với cả hai bản gốc.

## 3. Điều tra bản gốc — những phát hiện quan trọng

1. **Recipe int8 của họ** (không có trong repo, đã **pin bitwise**):

   ```python
   quantize_dynamic(model, out, op_types_to_quantize=["MatMul"], per_channel=True,
                    weight_type=QuantType.QInt8, reduce_range=True)
   ```

   - Weight per-output-channel đối xứng **7-bit** (scale = absmax/64, clamp ±64), activation per-tensor uint8 động (`DynamicQuantizeLinear`).
   - Áp cho đúng **84 Linear backbone** (12 lớp × q/k/v/o/gate/up/down) + **5 Linear acoustic** (qkv/o_proj/ff_gate/ff_up/ff_down). Attention QKᵀ/AV, RMSNorm, RoPE, softmax giữ fp32.
   - **Bằng chứng:** chạy pipeline của ta ở chế độ `--export-only` (không train) tái tạo bản shipped **khác 0.0**: 301/301 initializers backbone trùng, weights+scales acoustic trùng bitwise, outputs giống hệt.

2. **Graph "fp32" của họ (`onnx_update`) có acoustic đã int8 sẵn** — hai graph shipped chỉ khác nhau backbone. Vì vậy "fp32 baseline" thực chất là fp32-backbone + int8-acoustic.

3. **Embeddings/lm_heads/xvec_proj nằm ngoài ONNX** trong `vieneu_v3_heads.npz` (numpy fp32, 52 MB) — không bị quantize, giống nhau ở mọi engine. Phần này chiếm tỷ trọng lớn thời gian/frame trong sinh thô (16 matmul 1024×768 mỗi frame bằng numpy), làm loãng khác biệt tốc độ khi đo end-to-end.

4. Checkpoint `update/model.safetensors` lưu **bf16**; graph fp32 = upcast bf16→fp32 (bitwise).

## 4. Phương pháp

### 4.1 QAT fake-quant (khớp arithmetic deploy)

- Swap class `nn.Linear` → `QLinear` (tương thích state-dict) trên đúng 89 Linear của `semantic_backbone` + `acoustic_decoder`.
- Fake-quant weight per-channel 7-bit + activation per-tensor uint8 động, gradient thẳng (STE) — **trùng khớp phép toán mà `DynamicQuantizeLinear` + weight int8 sẽ làm trong ONNX Runtime**, nên cái train được chính là cái chạy.
- Sau train: **bake** — snap weights vào grid (requantize không mất mát) → `qat_model.safetensors`.

### 4.2 Export không cần re-export torch backbone

- Cấy thẳng weights vào graph fp32 gốc: 84 initializer ẩn `onnx::MatMul_*` khớp **bitwise** (lưu chuyển vị) với `semantic_backbone.*`; **tên initializer khác nhau giữa prefill và decode_step** → mapping từng file (`mapping.json`).
- Acoustic phải **tái export** từ mirror functional của `cached_step` (RMSNorm thủ công vì `aten::rms_norm` không export được); cần `requires_grad_(False)` + `no_grad` + `_promote_constants()` (nếu không weights bị fold vào Constant và `quantize_dynamic` bỏ qua → graph cứ nằm fp32).
- Quantize lại bằng đúng recipe pin ở mục 3.

### 4.3 Đóng gói (slim)

`quantize_dynamic` nhúng weights **inline từng graph** → prefill + decode_step mỗi file 1 bản copy ~104 MB (269 MB tổng). Fix: externalize weights prefill vào một `vieneu_backbone_shared.data`, map **theo giá trị bytes** → decode_step trỏ cùng offset. Verify chạy cùng input qua cả 2 bản (25 outputs × 2 graph) → **max|Δ| = 0.00e+00 (bitwise)** rồi mới thay thư mục.

## 5. Huấn luyện (tham số thật, lưu trong `qat1/train_args.json`)

| Tham số | Giá trị |
|---|---|
| Data | `train_2500.parquet` (dograh, 2.389 mẫu; eval 71 mẫu tách theo seed 42) |
| Train được | 89 Linear weights = **110.9 M / 124.0 M** params |
| Optimizer | AdamW β=(0.9, 0.98), lr 5e-5, wd 0.01, cosine + warmup 5%, clip 1.0 |
| Batch | 4 × grad-accum 4; 2 epochs → **290 bước (~3 phút GPU)**, bf16 autocast |
| Loss | CE text:audio = 1:8 (`vieneu_lora.compute_loss`) |

**Control experiment** (`--no-fake-quant`, run `ctrl1`): cùng data/seed/LR/steps nhưng forward là fp32 thuần (không fake-quant), không bake → export ra **PTQ của weights đã finetune**. Mục đích: tách "STE của QAT giảm sát thương quantize" khỏi "finetune tình cờ làm weights dễ quantize".

## 6. Kết quả

### 6.1 Teacher-forced (đo đúng sát thương quantize — metric đáng tin nhất)

Cùng 71 mẫu eval, input cố định. "fp32 twin" = cùng weights, không quantize → chênh lệch giữa twin và int8 **chính là sát thương quantize**.

| Cấu hình | audio CE | Chi phí quantize | acc cb0 (int8) | acc16 (int8) |
|---|---|---|---|---|
| base fp32 (weights gốc, không quantize) | 4.958 | — | 0.191 | 0.091 |
| base PTQ (weights gốc + quantize) — *int8 của họ* | 5.061 | +0.103 | 0.195 | 0.086 |
| **ctrl fp32-twin** (finetune fp32 thuần) | 4.162 | — | 0.394 | 0.147 |
| **ctrl PTQ** (finetune fp32 thuần → quantize) | 4.275 | **+0.114** | 0.352 | 0.138 |
| QAT fp32-twin | 4.185 | — | 0.394 | 0.146 |
| **QAT int8 (fake-quant)** | **4.241** | **+0.056** | **0.369** | **0.141** |

Top-1 agreement vs fp32 twin cùng weights: base-PTQ 59.3% · ctrl-PTQ 61.3% · **QAT 64.4%**.

Ba kết luận rút ra:

1. **Finetune thuần không giảm sát thương quantize** (+0.103 → +0.114, thậm chí nhích lên nhẹ) — weights sau finetune không "dễ quantize hơn" magical nào cả. Vậy mức **−46% của QAT là công dụng thật của fake-quant STE**, không phải tác dụng phụ của việc train.
2. **So sánh công bằng "cùng được finetune trên cùng data"** (dòng ctrl-PTQ vs QAT-int8): QAT thắng ở mọi chỉ số deploy (CE 4.241 vs 4.275; acc_cb0 0.369 vs 0.352; acc16 0.141 vs 0.138; agreement 64.4% vs 61.3%) — dù ctrl fp32-twin xuất phát tốt hơn một chút (4.162 vs 4.185). QAT "đánh đổi" chút ít chất lượng fp32 để mua nhiều độ bền int8.
3. Lượng tăng CE tuyệt đối so fp32 gốc (4.96→4.24) chủ yếu đến từ finetune trên dograh, phần quantize chỉ +0.056.

### 6.2 Nhiễu hidden state trong graph ONNX (feed seeded giống hệt, int8 vs fp32 twin)

| Graph | base PTQ relFro | ctrl PTQ relFro | QAT relFro |
|---|---|---|---|
| decode_step | 3.25% | 2.94% | **1.85% (−43%)** |
| prefill | 5.06% | 5.09% | **4.55% (−10%)** |
| acoustic | 1.1–1.8% | ~ngang | ~ngang |

Control cũng xác nhận ở mức graph: finetune fp32 thuần không làm giảm nhiễu quantize (2.94%/5.09% ≈ base PTQ).

### 6.3 Weight-space

| | PTQ | QAT |
|---|---|---|
| Lượng tử lại weights bị lệch (rel Fro, trung bình) | 1.66% | **0.000% (grid-exact sau bake)** |
| Drift so weights gốc | 0% (giữ nguyên) | 2.41% |

### 6.4 Free-running (sinh tự do, greedy) — **metric nhiễu, chỉ tham khảo**

| Lần chạy | Engine | frame-agree vs fp32 chung | EOS agree | RTF |
|---|---|---|---|---|
| 8 text | ptq | 39.3% | 62.5% | 1.20× |
| 8 text | qat | 29.8% | 75.0% | 1.27× |
| 6 text (khác) | ptq | 18.7% | 83.3% | — |
| 6 text (khác) | qat | 15.2% | 66.7% | — |
| So twin riêng (8 text) | qat-int8 vs qat-fp32 | 37.3% | 87.5% | 1.14× vs 1.35× |

Hai lần chạy **đảo chiều nhau** ở cả frame-agree lẫn EOS: greedy autoregressive compound (lệch 1 token → phần sau lệch hết), cả hai bản đều bimodal (text dễ 76–90%, text khó <3%). Khác biệt giữa hai bản int8 ở metric này nằm trong noise — hãy dựa vào 6.1–6.3.

**Nghịch lý "nói xong nhưng chưa kết thúc" trong eval greedy:** ở chế độ argmax thuần, sau khi hết lời model sinh **im lặng tuyệt đối** (RMS = 0.00) đến hết cap 180 frame vì EOS không bao giờ là argmax. Đây là artifact của protocol eval, không phải lỗi model: engine production decode bằng **sampling** (temperature 0.8, top_k 25, top_p 0.95) + **repetition_penalty 1.2** + trần frame theo độ dài phoneme (`frame_cap`) + `babble_retries`. Kiểm chứng bằng đúng `OnnxV3LiteEngine.infer` (sampling mặc định deploy): cả 3 engine dừng sạch ở 1.0–5.4 s đúng theo độ dài text (wav tại `engine_*/`), kèm tốc độ end-to-end thật: int8 ~2.5 s để sinh ~5 s audio vs fp32 ~4.2 s (**~1.7× nhanh hơn cả khi tính codec**).

### 6.5 Tốc độ (CPU 16 luồng; min của 3 vòng chạy process-riêng — trung vị/mean bị noise hệ thống ±20%)

| Op | fp32 gốc | PTQ int8 | QAT int8 | QAT nhanh hơn fp32 |
|---|---|---|---|---|
| prefill (T=64) | 26.9 ms | 15.6 ms | **15.3 ms** | 1.76× |
| decode_step (P=128) | 10.05 ms | 5.23 ms | **5.15 ms** | 1.95× |
| chuỗi acoustic (16 calls) | 3.57 ms | 4.34 ms | **2.59 ms** | 1.38× |
| ≈ 1 giây audio | 176 ms | 123 ms | **100 ms** | **1.76×** |

- QAT ≈ PTQ ở prefill/decode (graph cấu trúc giống hệt — đúng như lý thuyết, QAT chỉ đổi giá trị weights); acoustic của ta re-export gọn hơn nên nhanh hơn cả shipped (2.59 vs 4.34 ms).
- RTF end-to-end free-running chỉ chênh 1.37×→1.27× vì phần heads numpy (fp32, chung mọi engine) chiếm tỷ trọng lớn; codec audio không tính trong RTF.
- Benchmark tuần tự lần đầu cho thấy "QAT chậm hơn PTQ 15%" — xác định là **noise hệ thống** (thermal/nhiễu nền), loại bỏ bằng phương pháp đo ở trên.

### 6.6 Dung lượng & đóng gói

| Thư mục | Trước slim | Sau slim | Bản shipped tương ứng |
|---|---|---|---|
| QAT int8 | 269 MB | **158 MB** | 158 MB (bằng đúng) |
| QAT fp32-twin | 911 MB | **474 MB** | 454 MB (+20 MB vì acoustic của ta giữ fp32 thật, của họ nén int8 sẵn) |

Cấu trúc cuối giống hệt shipped: `vieneu_backbone_shared.data` 104 MB (1 bản, 2 graph trỏ chung) + 2 graph ~1.1 MB mỗi file + acoustic 7.2 MB + heads npz 52 MB. Outputs **bitwise không đổi** sau slim.

### 6.7 Phán quyết deploy: fp32-FT vs finetune→PTQ vs QAT (cả ba cùng finetune dograh)

Free-running 8 text (baseline = fp32 finetune; RTF của lần chạy này bị codec decode xen vào nên chỉ dùng bench cố định hình dạng ở 6.5 làm số tốc độ):

| | **fp32 finetune** | **finetune → PTQ** | **QAT int8** |
|---|---|---|---|
| Teacher-forced audio CE | **4.162** | 4.275 | 4.241 |
| acc_cb0 / acc16 (int8 deploy) | 0.394 / 0.147 | 0.352 / 0.138 | **0.369 / 0.141** |
| Mất mát so fp32 twin cùng weights | — | +0.114 | **+0.056** |
| Frame-agree vs fp32-FT (free-running) | baseline | 46.8% | 44.8% (ngang, trong noise) |
| EOS agree | 100% | 75.0% | **87.5%** |
| Tốc độ ≈1 s audio (bench 6.5) | 185 ms | 104 ms | **100 ms** |
| Dung lượng | 474 MB | 257 MB (slim → 158 MB) | **158 MB** |

**Kết luận:** chọn int8 thì chọn QAT (thắng PTQ ở mọi metric chất lượng, cùng tốc độ/size). Đổi int8 lấy ~0.08 CE so fp32-FT để được nhẹ 3× + nhanh 1.8×. Chênh lệch CE 4.162 vs 4.241 khá nhỏ — nghe thử `ctrl1/wav_ab/` để quyết định cuối.

**Head-to-head QAT int8 vs int8 shipped ở chế độ deploy** (sampling production, cùng seed RNG mỗi text → cùng chuỗi draws, khác biệt thuần do weights; 12 text × speaker thật): cả hai dừng sạch tự nhiên (không babble), thời lượng trung bình 3.91 s vs 4.24 s, **RTF 0.51× vs 0.53×** (tính cả codec). 0/12 text sinh waveform giống hệt nhau — kỳ vọng thuần thống kê: một frame = 16 code, per-code agreement ~60–64% ⇒ P(frame đầu toàn vẹn) ≈ 0.6¹⁶ ≈ 1/3000, sampling khuếch đại mọi khác biệt weights lên audio. Nghĩa là dưới sampling hai bản *khác nhau ở mức nghe được*; bản nào *hay hơn* chỉ tai người/ASR phán được (wav A/B tại `engine_ab/`).

## 7. Hạn chế & việc chưa làm

- **Train + eval đều trên dograh** (71 mẫu eval): mức tăng chất lượng đo trong domain/giọng này; chưa kiểm chứng tổng quát hóa ra giọng/ngôn ngữ khác (backbone drift 2.41%, heads không đổi).
- Chưa có **nghe thử bằng tai có kiểm soát** (wav đã render, xem mục 8); chưa đo WER/ASR-based.
- Bước slim nằm ở `/tmp/slim.py` (script ad-hoc, mất khi reboot) — **chưa gộp** vào `qat_v3turbo.py`; các script đo chi tiết (`detail_ce.py`, `detail_onnx.py`, `bench3.py`) cũng ở `/tmp`.
- Chỉ train 290 bước/2 epochs (mỗi bước ~0.6 s GPU) — dư dả không gian train thêm. int4 cũng 290 bước (0.67 s/bước, fake-quant blockwise tốn hơn chút).
- int4: chưa thử block_size 64/256 và symmetric mode; chưa đo int4 trên GPU (MatMulNBits có kernel CUDA, tốc độ/kết luận CPU chưa chuyển đổi trực tiếp được).
- Script đo int4 (`perturb4.py`, `twin4_ce.py`, `bench3.py`, `headtohead4.py`) nằm ở `/tmp` — mất khi reboot.

## 8. Artifacts

```
finetune/qat_v3turbo.py                  # pipeline QAT: attach→train→bake→export→quantize→self-check
finetune/eval_qat.py                     # eval greedy agreement + RTF + render wav
finetune/QAT_REPORT.md                   # báo cáo này
finetune/output/qat/                     # (gitignored)
  reference/onnx_update/  reference/onnx_int8/   # bản gốc copy làm chuẩn
  qat1/onnx_qat_int8/     # 158 MB — deploy được ngay
  qat1/onnx_qat_fp32/     # 474 MB — fp32 twin để đối chứng
  ctrl1/                  # control: finetune fp32 thuần → PTQ (onnx_qat_int8 257M chưa slim)
  ctrl1/wav_ab/           # 24 wav: ctrlfp32 + ctrlint8 + qat (8 text × 3) ← A/B "cả ba cùng finetune"
  engine_*/               # wav sinh bằng ĐÚNG engine production (sampling 0.8/rep 1.2), 3 graph × 3 text
  engine_ab/              # head-to-head production decode: shipped_{00..11} vs qat_{00..11}, cùng seed
  qat1/qat_model.safetensors  (473 MB, weights đã bake)
  qat1/mapping.json  qat1/train_args.json
  qat1/wav/       # 16 wav: qat + qat-fp32 twin (8 text × 2)
  qat1/wav_ab/    # 18 wav: fp32 + ptq + qat (6 text × 3) ← A/B tai tốt nhất
  qat4/onnx_qat_int4/     # 124 MB — QAT int4 deploy được ngay (mục 10)
  qat4/onnx_qat_fp32/     # 497 MB — fp32 twin int4 (CE 4.1753 = deployed)
  qat4/qat_model.safetensors (473 MB, weights đã bake grid int4)
  qat4/engine/            # 3 wav sinh bằng engine production qat4
  qat4/engine_ab/         # 24 wav head-to-head production: qat8_XX vs qat4_XX
  ptq4/onnx_qat_int4/     # 124 MB — baseline PTQ int4 (base weights, export-only)
  qat2/onnx_qat_int2/     # 92 MB — QAT int2 deploy được (mục 11; CHẬM 14× trên CPU)
  qat2/qat_model.safetensors (473 MB, weights đã bake grid int2)
  qat2/engine_ab/         # 24 wav head-to-head production: qat4_XX vs qat2_XX
  ptq2/onnx_qat_int2/     # 92 MB — baseline PTQ int2 (đã slim; fp32 twin + safetensors đã xóa cho đỡ đầy đĩa)
  listen/                 # 16 wav nghe thử 4 engine: {1..4}_{fp32goc,int8goc,qat8,qat4}.wav (cùng seed từng câu)
  listen_dograh/          # 20 wav như listen/ nhưng giọng dograh: {1..4}_{fp32goc,int8goc,qat8,qat4,qat2}.wav
  test_fpt/               # 4 wav qat4 câu call-center FPT (yêu cầu riêng; gồm 1fix = bản sửa lỗi gõ "hô trợ")
```

## 9. Tái lập

```bash
# 1) QAT train + export (mặc định = cấu hình mục 5, ~3 phút GPU)
.venv/bin/python -I finetune/qat_v3turbo.py \
    --data finetune/dataset/dograh_tts_dataset/train_2500.parquet --run qat1

# 1b) control: finetune fp32 thuần (không fake-quant) → PTQ của weights đã train
.venv/bin/python -I finetune/qat_v3turbo.py --run ctrl1 --no-fake-quant \
    --data finetune/dataset/dograh_tts_dataset/train_2500.parquet

# 1c) sanity: pipeline không train phải ra output giống hệt int8 shipped
.venv/bin/python -I finetune/qat_v3turbo.py --export-only --run ptqcheck \
    --data finetune/dataset/dograh_tts_dataset/train_2500.parquet

# 1d) QAT int4 (block 32) + PTQ-int4 baseline — mục 10
.venv/bin/python -I finetune/qat_v3turbo.py --run qat4 --bits 4 --block-size 32 \
    --data finetune/dataset/dograh_tts_dataset/train_2500.parquet
.venv/bin/python -I finetune/qat_v3turbo.py --export-only --run ptq4 --bits 4 --block-size 32 \
    --data finetune/dataset/dograh_tts_dataset/train_2500.parquet

# 1e) QAT int2 + PTQ-int2 baseline — mục 11
.venv/bin/python -I finetune/qat_v3turbo.py --run qat2 --bits 2 --block-size 32 \
    --data finetune/dataset/dograh_tts_dataset/train_2500.parquet
.venv/bin/python -I finetune/qat_v3turbo.py --export-only --run ptq2 --bits 2 --block-size 32 \
    --data finetune/dataset/dograh_tts_dataset/train_2500.parquet

# 2) eval + render wav A/B
.venv/bin/python -I finetune/eval_qat.py \
    fp32=finetune/output/qat/reference/onnx_update \
    ptq=finetune/output/qat/reference/onnx_int8 \
    qat=finetune/output/qat/qat1/onnx_qat_int8 \
    --n-texts 6 --wav-dir finetune/output/qat/qat1/wav_ab
```

## 10. QAT int4 — weight-only MatMulNBits (bổ sung cùng ngày)

### 10.1 Con đường triển khai

- int4 = **weight-only blockwise** qua contrib op `MatMulNBits`: weight packed uint4 + scale fp32 + zero-point 4-bit theo từng block 32 phần tử dọc input-dim; **activation giữ fp32** (khác int8 dynamic — quantize cả weight lẫn activation).
- Quantizer offline: `onnxruntime.quantization.matmul_nbits_quantizer.MatMulNBitsQuantizer` + `DefaultWeightOnlyQuantConfig(block_size=32)` (asymmetric mặc định). Không import được từ `onnxruntime.quantization`; cần `pip install onnx-ir`; gọi `q.model.save_model_to_file(...)` (không phải `q.save_...`); phải `logging.disable(logging.INFO)` nếu không muốn log 20k dòng.
- Chọn block_size **32**: probe PTQ trên decode_step thật cho relFro 10.97% (bs32) vs 15.15% (bs128); bs lớn chỉ tiết kiệm ~3% dung lượng (scales thưa hơn) nhưng hỏng hơn rõ.

### 10.2 Pin bitwise arithmetic kernel (MLAS) — phần khó nhất

Kernel thật nằm trong `onnxruntime/core/mlas/lib/q4_dq.cpp` (`BlockwiseQuantizer`, gọi qua pybind `quantize_matmul_4bits`). Công thức asymmetric, mỗi block bs phần tử dọc input-dim của từng out-channel:

```
mn = min(block, 0);  mx = max(block, 0)      # range LUÔN chứa 0
s  = (mx − mn) / 15                          # fp32
zp = clamp(roundf(−mn / s), 0, 15)           # CHIA thẳng cho s; roundf = nửa-xa-0
q  = clamp(roundf(w · (1/s) + zp), 0, 15)    # NHÂN reciprocal 1/s — không phải w/s
deq = (q − zp) · s
```

Ba chi tiết in đậm **vô hình trên dữ liệu random** (bản nháp đầu pin bitwise pass ngay trên ma trận ngẫu nhiên) nhưng trên weights thật — bf16-derived, thương số hay rơi đúng biên .5 — làm sai ~0.17% codes nếu mirror sai từng chi tiết. Sau khi đọc source C++ và sửa đúng: **bitwise 100%** (q + scales + zp) trên 13 weights thật × bs 32/128. Trong torch, `roundf` hiện thực bằng `_roundf` (nhảy qua float64 — `x + 0.5` exact trong double nên tie luôn đúng).

Giới hạn còn lại (chấp nhận, đã đo): bake lên grid rồi quantize lại khi export **không idempotent tuyệt đối** — ~0.07% codes lệch 1 mức, sai số deploy-vs-bake tối đa **0.5 bước grid**. Vô hại trong thực tế: self-check end-to-end int4 graph vs torch fake-quant cho max|Δ| = 4.5e-07 (mức noise thứ tự fp32).

### 10.3 Kết quả (cùng data/steps/eval-split như int8: dograh, 290 bước, seed 42)

| Metric | fp32 gốc | PTQ int4 (base weights) | **QAT int4** | QAT int8 (tham chiếu) |
|---|---|---|---|---|
| Audio CE teacher-forced (deployed) | 4.9583 | 4.9885 | **4.1753** | 4.241 |
| Chi phí quantize CE (vs fp32 cùng weights) | — | +0.0302 | **+0.0000** | +0.0556 |
| Hidden relFro decode_step (vs twin riêng)* | — | 23.27% | **0.01%** | 8.00% (PTQ int8: 9.73%) |
| Greedy frame-agreement vs fp32 twin | — | 17.0% | **93.1%** | 37.3% |
| EOS greedy (12 text) | — | 58.3% | **100%** | 87.5% |

\* harness mới (context lái bởi fp32 reference của từng cặp, 30 bước × 5 text thật); harness mục 6.2 cho int8 là 3.25%/1.85% — khác cách lái context, cùng chiều kết luận.

Ba điểm đáng chú ý:

- **Chi phí quantize = 0.0000**: sau bake, weights nằm trên grid int4 nên fp32 twin và bản deployed có CE bằng nhau đến 4 chữ số (4.1753/4.1753) — QAT "hấp thụ" trọn lượng tử hóa weight. int8 không đạt mức này vì activation còn bị quantize động lúc inference (dải động phụ thuộc input, training không bù hết được).
- **PTQ int4 phá free-running thảm khốc** (agreement 17%, EOS 58%) dù gap CE chỉ +0.030 — CE teacher-forced không nhìn thấy tổn thương compound autoregressive. Nghịch lý so với int8 (PTQ int8 gap CE lớn hơn nhưng free-running đỡ hỏng hơn): lỗi weight int4 (16 mức/block) lớn hơn nhiều lần lỗi weight int8 (128 mức/channel).
- QAT int4 CE 4.1753 < QAT int8 4.241: weight-only không phá activation nên finetune bù được gần hết.

### 10.4 Tốc độ (CPU 16 luồng, min 3 vòng process-riêng — cùng phương pháp 6.5)

| Op | fp32 gốc | int8 gốc (PTQ) | QAT int8 | PTQ int4 | **QAT int4** |
|---|---|---|---|---|---|
| prefill (T=64) | 26.0 ms | 17.8 ms | 16.1 ms | 21.8 ms | 21.9 ms |
| decode_step (P=128) | 10.3 ms | 5.70 ms | 5.25 ms | 4.69 ms | **4.70 ms** |
| chuỗi acoustic (16 calls) | 3.5 ms | 4.4 ms | 2.6 ms | 3.0 ms | 3.3 ms |
| ≈ 1 giây audio | 199 ms | 144 ms | **114 ms** | 118 ms | 122 ms |

- **decode_step (vòng nóng) int4 nhanh hơn mọi bản int8** (4.70 vs 5.25 QAT / 5.70 gốc).
- Cả 2 engine của ta **thắng cả int8 gốc end-to-end** (114/122 vs 144 ms) — acoustic graph của ta re-export gọn hơn nhiều (2.6–3.3 vs 4.4 ms của shipped).
- Prefill + acoustic int4 chịu overhead dequant MatMulNBits → ≈1s audio chậm hơn QAT int8 ~7% trên CPU này; càng audio dài càng có lợi cho int4 (decode chiếm tỷ trọng lớn).
- **Production end-to-end** (engine thật, sampling, 12 text, cùng seed mỗi text): **RTF 0.34× cho cả hai** QAT, thời lượng trung bình 3.91 s vs 3.91 s — phần trừng phạt prefill bị san phẳng ngoài thực tế. 0/12 waveform giống hệt (kỳ vọng thống kê như mục 6.7). Wav A/B: `qat4/engine_ab/` (qat8_XX vs qat4_XX).
- **Nghe thử 4 engine** (fp32 gốc / int8 gốc / QAT int8 / QAT int4, cùng seed từng câu, text→speech đầy đủ qua G2P sea_g2p): `listen/{1..4}_{fp32goc,int8goc,qat8,qat4}.wav` — 4 câu ngắn→trung có câu đọc số. RTF production 4 câu này: fp32 0.53–0.63×, int8 gốc 0.37–0.52×, qat8 0.48–0.68×, qat4 0.48–0.58× (trong noise nhiệt; các bản quantized ngang nhau, fp32 chậm nhất).
- **Nghe thử giọng dograh** (`listen_dograh/{1..4}_{fp32goc,int8goc,qat8,qat4}.wav`): cùng 4 câu + seed như listen/, nhưng thay voice preset bằng speaker_embedding centroid của cụm cos>0.85 (1716/2389 mẫu → embedding 1 người nói rất ổn định). RTF 0.33–0.47×, qat4 nhanh nhất cả 4 câu. **Đọc kết quả**: khác biệt giữa `listen/` vs `listen_dograh/` (cùng engine) là *giọng nói*; khác biệt *gốc vs qat* trong `listen_dograh/` là finetune + quantize *cộng dồn* (fp32 gốc / int8 gốc chưa từng học giọng dograh — zero-shot cloning, còn qat8/qat4 được finetune trên chính giọng này), nên bản finetuned được kỳ vọng tự nhiên hơn chứ không chỉ "khác vì quantize".

### 10.5 Dung lượng & đóng gói

| Thư mục | Trước slim | Sau slim |
|---|---|---|
| QAT int4 | 191 MB | **124 MB** |
| PTQ int4 (baseline) | 191 MB | 124 MB |
| QAT int4 fp32-twin | 912 MB | 497 MB |

Cấu trúc giống int8: `vieneu_backbone_shared.data` 63.5 MB (1 bản dùng chung, 277 blobs dedupe, verify **bitwise 0.00**) + graph 0.3 MB × 2 + acoustic 4.4 MB + heads npz 49.8 MB. So int8 (158 MB): **−34 MB (−21%)**; so fp32 gốc (454 MB): −73%.

### 10.6 Phán quyết int4 vs int8 (deploy)

- Chất lượng: QAT int4 ≥ QAT int8 trên mọi metric đo được.
- Dung lượng: nhỏ hơn 21%.
- Tốc độ: decode nhanh hơn 10%; end-to-end production ngang bằng (RTF 0.34× cả hai).
- **PTQ int4 loại khỏi consideration** — int4 chỉ khả thi sau QAT.

→ Trên CPU cả hai đều hợp lệ: **qat4 nếu ưu tiên size + chất lượng, qat8 nếu ưu tiên throughput bench hình dạng cố định**. Deploy GPU sẽ khác (MatMulNBits có kernel CUDA riêng). Cách deploy giống hệt, chỉ đổi `onnx_dir`:

```python
Vieneu(mode="v3turbo", onnx_dir="finetune/output/qat/qat4/onnx_qat_int4")
```

## 11. QAT int2 — 2-bit, giới hạn dưới của weight-only (bổ sung 2026-10-24)

### 11.1 Con đường triển khai — khả thi về nguyên tắc

`MatMulNBitsQuantizer` chấp nhận `bits ∈ {2,4,8}`; pybind ORT 1.31 có sẵn `quantize_matmul_2bits` — **cùng template C++ `QuantizeMatMulNBitsBlockwise<T, qbits>` với 4-bit**, tức arithmetic đã pin ở mục 10.2 giữ nguyên, chỉ thay `levels = 2²−1 = 3` (`s=(mx−mn)/3`, `zp,q ∈ 0..3`). Verified 2 lớp:
- mini-model MatMul bits=2 chạy CPU EP, đối chiếu mirror torch levels=3: max|Δ| **5.96e-07** (bitwise theo fp32);
- pipeline thật `qat_v3turbo.py --bits 2` self-check prefill: max|Δ| **2.01e-07**.

Script tổng quát hóa `_quant_nbits(wb, levels)`; `--bits 2` chạy đủ train/export/self-check.

### 11.2 Kết quả (cùng data/steps/eval-split: dograh, 290 bước, seed 42)

| Metric | PTQ int2 | **QAT int2** | QAT int4 (đối chiếu) |
|---|---|---|---|
| Hidden relFro decode_step | **91.79%** (vs fp32 gốc) | **0.03%** (vs twin riêng) | 0.01% |
| Greedy agreement vs twin | 1.7% | **69.5%** | 93.1% |
| EOS greedy | 58.3% | **83.3%** | 100% |
| Twin audio CE (71 row) | — | **4.4574** | 4.1753 |
| Deployed CE (= twin) | — | 4.4574 (**quant cost +0.0000**) | 4.1753 (+0.0000) |
| Gap CE khởi đầu (chưa train) | — | **+0.956** | +0.030 |

- QAT int2 train hội tụ nhanh (gap +0.956 vá xong từ step ~20), CE cuối **4.4575** — dưới fp32 gốc 4.9583; acc_cb0 0.401 (fp32 0.243).
- **Trần chất lượng thấp hơn int4 đúng +0.28 CE** — 3 mức/block buộc fp32 twin hội tụ về optimum kém hơn; không phải sát thương deploy (quant cost vẫn +0.0000 vì weights nằm sẵn trên grid).
- Head-to-head production 12 text cùng seed: dur trung bình 3.92s (qat4 3.82s), **0/12 waveform giống nhau, t_div 0.00s toàn bộ** (khác ngay frame đầu — hai engine là hai model khác nhau về bản chất).
- Wav A/B: `qat2/engine_ab/` (qat4_XX vs qat2_XX), nghe thêm `listen_dograh/{1..4}_qat2.wav`.

### 11.3 Tốc độ — tử huyệt của int2 trên CPU

Bench cùng phương pháp 6.5 (min 3 vòng, process riêng, CPU 16 luồng):

| Graph | fp32 gốc | int8 gốc | QAT int8 | QAT int4 | **QAT int2** |
|---|---|---|---|---|---|
| prefill | 10.9 | 17.8 | 16.1 | 31.3 | **80.1** |
| decode_step | 10.3 | 5.70 | 5.25 | 4.99 | **71.1** |
| acoustic | 4.4 (shipped) | — | 2.6 | 3.7 | **71.1** |

- decode_step int2 chậm hơn int4 **14×**, chậm hơn cả fp32 7×. Production engine thật (3 text): RTF **2.05–2.30×** (chậm hơn realtime; qat4: 0.33×); head-to-head 12 text: gen 2.2–11.9s/text.
- **Cơ chế chính xác (đọc source `contrib_ops/cpu/quantization/matmul_nbits.cc` @v1.31.0)**: chỉ bits=4 có đường packed assembly `MlasQ4GemmBatch`; bits=2 (dòng 1222–1240) **dequantize TOÀN BỘ weight matrix về fp32 buffer tạm MỖI LẦN CHẠY rồi GEMM fp32 thường** (source còn để TODO "move B dequant to prepack"). Decode_step có 84 MatMul → mỗi frame nộp lại thuế dequant cả backbone; acoustic graph cũng bị quantize int2 theo pipeline nên gấp đôi thiệt hại (71ms + 71ms mỗi frame).
- Lưu ý acoustic graph cũng bị quantize theo (`quantize_dir` chạy cả 3 graph) — 71ms/frame acoustic là cùng một hiện tượng.

### 11.3.1 Fix: expand-at-load — dequant 1 lần lúc load (đã làm, validate + đo)

Vì kernel bits=2 cuối cùng cũng tính bằng fp32 GEMM trên weights đã dequant, ta tự dequant **một lần lúc load** thay vì mỗi frame: unpack codes (cặp bit little-endian, 4 codes/byte) + zp (4 zp/byte) → `W=(q−zp)·s` fp32 → thay node `MatMulNBits` bằng `MatMul` thường, tạo session **từ bytes trong RAM** (đĩa vẫn giữ file int2 92MB). Validate prefill vs graph int2 gốc: max|Δ| **1.49e-07** (mức fp32 — cùng arithmetic).

| ms (CPU 16T, min reps) | int2 thô | **int2 expand-at-load** | fp32 twin | QAT int4 |
|---|---|---|---|---|
| prefill | 80.1 | **27.0** | ~11 | 31.3 |
| decode_step | 71.1 | **11.15** | 10.3 | 4.99 |
| acoustic | 71.1 | **3.47** | 3.3 | 3.7 |
| production RTF (3 text) | 2.05–2.30× | **0.54–0.61×** | ~0.45× | 0.33× |

→ int2 expand-at-load = **92 MB đĩa / ~450 MB RAM / tốc độ ngang fp32**. Script: `/tmp/expand2.py` (mất khi reboot; công thức unpack ghi ở trên).

### 11.4 Dung lượng & đóng gói

Slim dedupe như 10.5: **92 MB** (int4: 124 MB, −26%; int8: 158 MB; fp32: 454 MB). Verify bitwise 0.00. Cấu trúc: .data ~31.5MB (½ của int4) + graphs + acoustic + heads npz 49.8MB.

### 11.5 Phán quyết int2

- **Chất lượng: qua** — QAT giữ được int2 ở mức nghe được (agreement 69.5%, trần CE +0.28 vs int4), đối lập hoàn toàn với PTQ int2 (91.8% relFro = vỡ).
- **Tốc độ CPU dạng thô: loại** — 14× chậm hơn int4. Nhưng **fix expand-at-load (11.3.1) đưa về ngang fp32 (RTF 0.54–0.61×)** với đĩa vẫn 92MB — dùng được nếu ép buộc int2 (đĩa nhỏ, RAM dư).
- **Kết luận (cập nhật sau 11.6): điểm ngọt weight-only trên CPU chuyển thành int2 ≈ int4** — kernel MLAS Q2Gemm native (11.6) đưa decode về 5.3 ms, nhanh hơn cả expand-at-load 2× và không tốn RAM mở rộng; chờ được merge upstream.
- Wav: `qat2/engine_expand/` (3 wav production expand-at-load).

### 11.6 Kernel MLAS Q2Gemm native — viết & đóng góp ngược upstream ORT (2026-10-24)

Nguyên nhân gốc (đọc source ORT 1.31.0): `GetQNBitGemmVariant` chỉ trả variant packed cho bits=2 khi compute type là `SQNBIT_CompInt8`; với activation float (mặc định) = `SQNBIT_CompFp32` → không có variant → PrePack bỏ qua → mỗi lần chạy dequant toàn bộ B + SGEMM thường (đường 71 ms).

Đã viết patch (clone `~/workspace/ort-q2`, branch `q2bit-compfp32`, commit `d90a51e`, 9 files +645/−22):
- `SQ2BitGemmVariant_CompFp32` mới: kernel M==1 (decode, AVX2, unpack 32 trọng số/lần bằng 4-phase and/shift + interleave), dequant-B-to-SGEMM-pack cho M>1 (prefill), prepack = copy thuần (kernel ăn native layout của quantizer, không swizzle).
- Đăng ký vào cả 4 bảng dispatch x64 (AVX2 / AVX-VNNI / AVX-512 / AVX-512-VNNI); sửa `MlasQNBitZeroPointsForBlksSizeInBytes<2>` sai từ nb/2 → nb/4; generalize oracle CompInt8 của unit test theo bit-width → codepath W2 CompInt8 sẵn có của upstream lần đầu có unit test.

Kết quả trên model qat2 thật (92MB, asymmetric zp, đối chiếu bit với pip ORT 1.31.0 cùng graph):

| | pip ORT 1.31.0 (unpacked) | **custom build (kernel mới)** | expand-at-load | QAT int4 |
|---|---|---|---|---|
| decode_step | 61–73 ms | **5.3–5.5 ms (12–13×)** | 11.15 ms | 4.99 ms |
| prefill 64 tok | 109–131 ms | **25–38 ms (~3.5×)** | 27.0 ms | 31.3 ms |
| max\|Δ\| vs pip | — | **1.8e-06 / 5.2e-06 = PASS** | 1.49e-07 | — |
| RAM | 92 MB | **92 MB (không expand)** | ~450 MB | 124 MB đĩa |

- Unit test MLAS: 9371 test 2-bit (cả CompFp32 + CompInt8, M/N/K có tail, bias, threaded) **pass hết**; tổng suite SQNBitGemm 24.705/24.722 (17 skip do ARM KleidiAI).
- Bug thật bắt được nhờ validate model (unit test không bắt được vì `MatrixGuardBuffer` fill ramp xác định → zp đồng nhất mọi block → sai zp-per-column không lộ): lần đầu viết dequant kernel lấy zp của cột đầu áp cho cả nhóm 8 cột → prefill lệch 9.4; sửa theo đúng pattern upstream 4-bit (zp từng cột) → PASS.
- Int2 giờ **nhanh hơn fp32 gốc** (5.3 vs 10.3 ms decode) và chỉ chậm int4 ~6% (mặc dù unpack 2-bit đắt hơn trên byte) — nhờ trọng số nhỏ hơn một nửa.
- PR upstream: đã mở **#33573** lên microsoft/onnxruntime (2026-10-24, cherry-pick lên main tip, 9 files +645/−22, commit `82e683d7`) rồi **đóng theo yêu cầu** — giữ code ở fork riêng `github.com/Ripefog/onnxruntime`, branch `q2bit-compfp32` (draft PR text: `~/workspace/ort-q2/Q2_COMPFP32_PR_DRAFT.md`).

### 11.7 Khuyết tật "ngắt câu cuối" — EOS sớm, KHÔNG phải kernel/cap (2026-10-24)

Hiện tượng: wav câu địa chỉ "Nhà anh ở 91C/16 Ngô Chí Quốc, Tân Bình phải không ạ? Em cho kỹ thuật xuống kiểm tra ngay đây ạ" dừng ở 3.44s, mất hẳn câu 2. Chẩn đoán (đếm frames @12.5 fps bằng monkeypatch `_acoustic_frame`):

- **Không phải frame cap**: sinh 43 frame rồi EOS trong khi trần `max_expected_frames` = 352 — vòng lặp không chạm trần.
- **Không phải kernel mới**: kernel bit-exact vs pip ORT (1.8e-06); và fp32 twin cũng dính.
- **Cơ chế**: 2 câu (~100 ký tự) nằm chung 1 chunk (`normalize_to_chunks_v3`, max_chars 256 không cắt theo câu); sau câu hỏi 1 ("...phải không ạ?") là điểm dừng tự nhiên → đầu EOS dao động, sampling quyết định dừng hay tiếp. Phân phối frames lặp 4 lần/câu: fp32 {45,47,68,67}, int2 {42,44,67,71} — **lưỡng phân, fp32 cũng cắt ~50%**.
- int2 làm tệ hơn + lệch nội dung đơn câu: fpt3 int2 27 frame vs fp32 40 (khớp EOS agreement 83.3% lúc eval).
- Engine có guard "nói thêm" (`babble_suspect`) nhưng **không có guard "nói thiếu"** — khoảng trống thật sự.
- Xử lý tạm: re-roll đến khi frames ≥ 60 (thường đạt trong 1–2 lần) → `engine_q2kernel/fpt_addr_q2kernel.wav` (67f, 5.36s) + bản đối chiếu `fpt_addr_fp32ref.wav` (67f, 5.36s).
- Fix đề xuất (chưa làm): (a) min-frames retry guard ~1.8 frame/âm tiết đối xứng với babble guard; hoặc (b) ưu tiên cắt chunk ở ranh giới câu khi chunk chứa ≥ 2 câu.
## Phụ lục: gotchas kỹ thuật gặp phải

- `onnx.save_model(save_as_external_data=True)` **append** vào `.data` cũ và **mutate chính model proto** — phải dọn thư mục trước + snapshot bytes trước khi save.
- Xóa `raw_data` bằng `t.raw_data = b""` **vẫn tính là field present** → `onnx.save` cố append và crash; phải `t.ClearField("raw_data")`.
- `quantize_dynamic` nhúng weights inline từng graph (không external) và **bỏ qua weights bị fold vào Constant** — cần promote Constant→initializer trước khi quantize.
- safetensors `save_file` lỗi với tensor chia sẻ bộ nhớ (tied embeddings) → dùng `save_model` (lưu ý nó bỏ key tied, điền lại từ checkpoint gốc khi eval).
- Tên initializer `onnx::MatMul_*` **khác nhau giữa prefill và decode_step** → mapping từng file.
- `torch.Tensor.numpy()` fail với bf16 → `.float()` trước (upcast bitwise chính xác).
- **int4 (MatMulNBits)**: import `MatMulNBitsQuantizer` từ `onnxruntime.quantization.matmul_nbits_quantizer` (không có ở `onnxruntime.quantization`); cần `pip install onnx-ir`; save qua `q.model.save_model_to_file(...)`; `logging.disable(logging.INFO)` nếu không bị log nhấn chìm.
- **Arithmetic int4 phải pin trên weights thật, không phải random**: 3 chi tiết (range chứa 0, zp qua phép chia, q qua nhân reciprocal + roundf) đều vô hình trên random (pass bitwise giả) nhưng sai 0.17% codes trên weights bf16-derived vì thương số rơi đúng biên .5. Nguồn chân lý: `core/mlas/lib/q4_dq.cpp`.
- **Đo hidden-perturbation phải so với fp32 twin CỦA CẦU ĐÓ** — so với fp32 gốc thì phần lớn relFro là drift finetune (60.9%!) chứ không phải thiệt hại quantize.
- **int2 (bits=2)**: `MatMulNBitsQuantizer(model, algo_config=...)` phải truyền `algo_config` **bằng keyword** — tham số vị trí thứ hai là `bits` (truyền config vào vị trí đó → AssertionError "Only support 2, 4 or 8 bits" rất gây nhiễu).
- **onnx-ir ghi IR version 14 nhưng ORT 1.31 runtime chỉ load ≤ 13** → sau `save_model_to_file` phải hạ `ir_version` về ≤ 13 rồi `onnx.save` lại (chỉ gặp khi quantizer parse từ model IR mới).
- **Kernel CPU bits=2 không có assembly MLAS** (mlas_q4.h chỉ có Q4Gemm/Q8Q4Gemm) → decode chậm 14× int4; đo trước khi đầu tư train.
