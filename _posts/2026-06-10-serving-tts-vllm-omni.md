---
layout: post
title: "Serving TTS Models with vLLM-Omni"
author: "vLLM-Omni TTS Team"
image: /assets/figures/2026-06-10-serving-tts-vllm-omni/hero.png
tags:
  - multimodal
  - tts
  - vllm-omni
---

## TL;DR

<!-- Brief summary: what's new, which models are supported, key performance highlights (3-5 bullets). -->

vLLM-Omni now supports high-performance TTS serving for four models — Qwen3-TTS, VoxCPM2, Fish Audio, and Higgs Audio — with stage-based pipeline optimization, real-time streaming, async chunk scheduling, and CUDA graph acceleration.

- **Bullet 1:** Key highlight (e.g., VRAM savings, latency improvement, or model coverage)
- **Bullet 2:** Key highlight
- **Bullet 3:** Key highlight
- **Bullet 4:** Key highlight

## 1. Introduction

<!-- 2-3 paragraphs. Why TTS serving matters. The gap in existing serving solutions. How vllm-omni's architecture naturally extends to TTS. End with a roadmap of what this post covers. -->

## 2. Deep Dive: Qwen3-TTS

### 2.1 Model Overview

<!-- Describe Qwen3-TTS: architecture (AR-based), input/output modalities, model size, key specs. What makes it different from traditional TTS models? -->

### 2.2 Stage Abstraction

<!-- How vllm-omni decomposes Qwen3-TTS into pipeline stages. What are the stages (e.g., text encoder → token decoder → vocoder)? How are stages defined and composed? -->

<p align="center">
<img src="/assets/figures/2026-06-10-serving-tts-vllm-omni/qwen3-tts-stage-graph.png" width="80%">
<br>
<em>Figure 1: Stage decomposition of Qwen3-TTS in vLLM-Omni.</em>
</p>

### 2.3 Streaming Support

<!-- How chunk-level audio streaming works end-to-end for Qwen3-TTS. Client API for consuming streamed audio. Latency-to-first-chunk numbers if available. -->

### 2.4 Async Chunk Scheduling

<!-- How async_chunk works for Qwen3-TTS. How it overlaps computation across stages and requests. Parallelism opportunities in the AR architecture. -->

### 2.5 CUDA Graphs

<!-- How CUDA graphs are applied to Qwen3-TTS stages. Which stages benefit, capture strategy, warmup, performance gains. -->

### 2.6 Supported Modes

<!-- Serving modes: single-speaker, multi-speaker, voice clone, streaming vs. batch. How they map to stage configurations. -->

### 2.7 Quick Start

<!-- Actual serving command -->
```bash
vllm serve Qwen/Qwen3-TTS --omni --port 8091
```

<!-- Actual API call example (update model path and parameters) -->
```bash
curl -s http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS",
    "input": "Hello, welcome to vLLM-Omni TTS serving.",
    "voice": "Chelsie",
    "stream": true
  }'
```

## 3. Deep Dive: VoxCPM2

### 3.1 Model Overview

<!-- Describe VoxCPM2: architecture (diffusion-based), how it differs from Qwen3-TTS's AR approach, model size, key specs. -->

### 3.2 Stage Abstraction

<!-- How vllm-omni decomposes VoxCPM2 into pipeline stages. How the multi-step denoising loop maps to stages. Contrast with Qwen3-TTS's stage graph. -->

<p align="center">
<img src="/assets/figures/2026-06-10-serving-tts-vllm-omni/voxcpm2-stage-graph.png" width="80%">
<br>
<em>Figure 2: Stage decomposition of VoxCPM2 in vLLM-Omni.</em>
</p>

### 3.3 Streaming Support

<!-- How streaming works for VoxCPM2's diffusion architecture. Different chunking strategy vs AR models. Latency comparison with Qwen3-TTS. -->

### 3.4 Async Chunk Scheduling

<!-- Async chunk scheduling for VoxCPM2. Different parallelism opportunities in diffusion vs AR. How scheduling differs from Section 2.4. -->

### 3.5 CUDA Graphs

<!-- CUDA graph capture for diffusion stages. Different warmup/replay characteristics vs AR stages. Performance gains. -->

### 3.6 Supported Modes

<!-- Serving modes specific to VoxCPM2. -->

### 3.7 Quick Start

<!-- Actual serving command -->
```bash
vllm serve openbmb/VoxCPM2 --omni --port 8091
```

<!-- Actual API call example (update model path and parameters) -->
```bash
curl -s http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/VoxCPM2",
    "input": "Welcome to VoxCPM2 TTS serving with vLLM-Omni.",
    "stream": true
  }'
```

## 4. Quick Start: Fish Audio & Higgs Audio

### 4.1 Fish Audio

<!-- 2-3 sentences: what Fish Audio is, voice cloning / custom voice capabilities, notable features. -->

<!-- Serving command (update model path) -->
```bash
vllm serve fishaudio/fish-speech-1.5 --omni --port 8091
```

<!-- API call example if the API differs from standard speech endpoint -->

### 4.2 Higgs Audio

<!-- 2-3 sentences: what Higgs Audio is, music / audio generation capabilities, notable features. -->

<!-- Serving command (update model path) -->
```bash
vllm serve Higgs-Labs/higgs-audio --omni --port 8091
```

## 5. Model Comparison

<!-- Fill in all cells with accurate benchmark data and model specs -->

| | **Qwen3-TTS** | **VoxCPM2** | **Fish Audio** | **Higgs Audio** |
|---|---|---|---|---|
| **Architecture** | Autoregressive | Diffusion | — | — |
| **Streaming** | — | — | — | — |
| **Voice cloning** | — | — | — | — |
| **Multilingual** | — | — | — | — |
| **VRAM** | — | — | — | — |
| **Latency** | — | — | — | — |
| **Best for** | General TTS | High-quality TTS | Voice cloning | Music / audio |

## 6. Conclusion

<!-- 1-2 paragraphs summarizing what was covered. The key takeaway about vllm-omni's architecture handling diverse TTS models. -->

### Getting Started

```bash
pip install vllm-omni
vllm serve Qwen/Qwen3-TTS --omni --port 8091
```

### Resources

- **vLLM-Omni:** [github.com/vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)
- **Documentation:** [vllm-omni docs](https://github.com/vllm-project/vllm-omni#documentation) <!-- update with actual docs URL -->
- **Model cards:** [Qwen3-TTS]() · [VoxCPM2]() · [Fish Audio]() · [Higgs Audio]() <!-- add HuggingFace links -->
- **Community:** [Discord](https://discord.gg/vllm) · [GitHub Discussions](https://github.com/vllm-project/vllm-omni/discussions)

## Acknowledgements

<!-- List GitHub IDs of contributors to the TTS serving feature. Format: Special thanks to @handle1, @handle2, ... for their contributions to TTS model support in vLLM-Omni. -->
