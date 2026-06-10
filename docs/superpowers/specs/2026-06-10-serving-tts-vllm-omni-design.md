# Design: Serving TTS Models with vLLM-Omni Blog Post

**Date:** 2026-06-10
**Branch:** `blog/serving-tts-vllm-omni`
**Author attribution:** vLLM-Omni TTS Team (with acknowledgments to specific GitHub IDs)

## Overview

A developer-focused blog post for the vLLM blog (`vllm-project.github.io`) covering how to serve TTS models using vLLM-Omni. The post covers 4 models — Qwen3-TTS, VoxCPM2, Fish Audio, and Higgs Audio — with deep architectural dives on Qwen3-TTS and VoxCPM2, and brief surveys of Fish Audio and Higgs Audio.

## Audience

vLLM users and developers who want to serve TTS models. Practical, code-heavy, assumes familiarity with vLLM basics.

## Structure

### Section 1: TL;DR / Intro

- **TL;DR** bullet block summarizing what's new: TTS serving support across 4 models, key performance highlights
- **Why TTS in vLLM-Omni?** — Context on the growing importance of speech output in omni models, the gap in existing serving solutions (most engines are text-in-text-out), and how vllm-omni's architecture naturally extends to TTS

**Front matter:**
```yaml
layout: post
title: "Serving TTS Models with vLLM-Omni"
author: "vLLM-Omni TTS Team"
image: /assets/figures/2026-06-10-serving-tts-vllm-omni/<hero-image>
tags:
  - multimodal
  - tts
  - vllm-omni
```

### Section 2: Deep Dive — Qwen3-TTS

Full architectural walkthrough of how vllm-omni serves Qwen3-TTS internally:

1. **Model overview** — Qwen3-TTS architecture, input/output modalities
2. **Stage abstraction** — How vllm-omni decomposes Qwen3-TTS into pipeline stages (e.g., text encoder → token-to-speech decoder → vocoder), how stages are defined and composed
3. **Streaming support** — Chunk-level audio streaming end-to-end (client → API → stage pipeline → chunked response), enabling real-time playback before full generation completes
4. **async_chunk** — Async chunk scheduling within and across stages — overlapping computation to hide latency, how chunks flow through the pipeline
5. **CUDAGraph** — How CUDA graphs are applied to TTS stages — which stages benefit, capture strategy, warmup considerations, performance gains
6. **Supported modes** — Serving modes available (e.g., single-speaker, multi-speaker, voice clone, streaming vs. batch), how they map to stage configurations
7. **Code example** — Full serving command + API call example with streaming output

### Section 3: Deep Dive — VoxCPM2

Same structural treatment as Section 2, but for VoxCPM2. Key value: readers see vllm-omni's stage abstraction working across fundamentally different architectures (AR vs diffusion), reinforcing framework generality.

1. **Model overview** — VoxCPM2 architecture and how it differs from Qwen3-TTS
2. **Stage abstraction** — How vllm-omni decomposes VoxCPM2's pipeline (diffusion/flow-matching based), contrasting stage graph with Qwen3-TTS's
3. **Streaming support** — How streaming works for a non-AR / diffusion-based model (different chunking strategy)
4. **async_chunk** — Async scheduling for VoxCPM2's stages — different parallelism opportunities than AR models
5. **CUDAGraph** — CUDA graph capture for diffusion stages — different warmup/replay characteristics
6. **Supported modes** — Serving modes specific to VoxCPM2
7. **Code example** — Serving command + API call

### Section 4: Survey — Fish Audio & Higgs Audio

Brief coverage of remaining two models. Each gets:
- 1-2 paragraphs on what the model is and notable capabilities
- Quick serving command (code block)
- Link to model card

**Fish Audio:** Voice cloning / custom voice focus.
**Higgs Audio:** Music / audio generation focus.

Roughly 3-5 sentences + code block per model.

### Section 5: Comparison Table

Single table comparing all 4 models across practical dimensions:

| Dimension | Qwen3-TTS | VoxCPM2 | Fish Audio | Higgs Audio |
|---|---|---|---|---|
| Architecture | AR | Diffusion | ... | ... |
| Streaming | ... | ... | ... | ... |
| Voice clone | ... | ... | ... | ... |
| Multilingual | ... | ... | ... | ... |
| VRAM (approx.) | ... | ... | ... | ... |
| Latency | ... | ... | ... | ... |
| Recommended use case | ... | ... | ... | ... |

### Section 6: Conclusion & Getting Started

- Summary of what was covered
- Links: vllm-omni repo, docs, model cards, Discord/community
- Acknowledgments section listing specific GitHub IDs
- Call to action (try it, open issues, contribute)

## File Layout

```
_posts/2026-06-10-serving-tts-vllm-omni.md
assets/figures/2026-06-10-serving-tts-vllm-omni/
  <hero-image>
  <architecture-diagrams>
  <comparison-chart>
```

## Conventions

- Follow existing vLLM blog style (Jekyll Minima theme, GFM markdown)
- Hero image in front matter `image` field
- Figures in `assets/figures/2026-06-10-serving-tts-vllm-omni/`
- Code blocks with language annotations
- MathJax available via `math: true` if needed
- GitHub-flavored admonitions supported (`> [!NOTE]`, etc.)
