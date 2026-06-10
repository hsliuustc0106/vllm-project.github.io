# Serving TTS Models with vLLM-Omni Blog Post — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a developer-focused blog post covering TTS model serving in vLLM-Omni, with deep architectural dives on Qwen3-TTS and VoxCPM2, plus brief surveys of Fish Audio and Higgs Audio.

**Architecture:** Single Jekyll markdown post with figures in a dedicated assets directory. Follows existing vLLM blog conventions (Minima theme, GFM markdown, numbered sections, code examples, centered figures).

**Tech Stack:** Jekyll, Markdown, HTML `<p align="center">` for figures, bash/python code blocks.

---

### Task 1: Create the figures directory and placeholder hero image

**Files:**
- Create: `assets/figures/2026-06-10-serving-tts-vllm-omni/` (directory)

- [ ] **Step 1: Create the figures directory**

```bash
mkdir -p assets/figures/2026-06-10-serving-tts-vllm-omni
```

Run from project root: `/Users/elle/claude-code/vllm-project.github.io`

- [ ] **Step 2: Commit**

```bash
git add assets/figures/2026-06-10-serving-tts-vllm-omni/.gitkeep
git commit -m "chore: create figures directory for TTS blog post"
```

Create a `.gitkeep` file so the empty directory is tracked by git.

---

### Task 2: Create the blog post file with front matter and Section 1 (TL;DR / Intro)

**Files:**
- Create: `_posts/2026-06-10-serving-tts-vllm-omni.md`

- [ ] **Step 1: Create the post file with front matter and intro**

Create `_posts/2026-06-10-serving-tts-vllm-omni.md` with the following content. The hero image path should match whatever image lands in the figures directory. Use a placeholder name that will be updated when the real image is added.

```markdown
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

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) now supports high-performance serving for text-to-speech (TTS) models, bringing vLLM's production-grade serving infrastructure to speech generation workloads. This post covers four TTS models — Qwen3-TTS, VoxCPM2, Fish Audio, and Higgs Audio — with deep architectural dives into how vllm-omni's stage abstraction, streaming, async chunk scheduling, and CUDA graphs make TTS serving fast and flexible.

Key highlights:

- **Unified TTS serving:** Serve multiple TTS architectures (AR, diffusion, voice cloning) through a single vLLM-Omni runtime.
- **Real-time streaming:** Chunk-level audio streaming enables playback before full generation completes.
- **Architecture-aware optimization:** Stage decomposition, async chunk scheduling, and CUDA graph capture deliver low-latency inference across diverse model architectures.
- **4 models, 1 framework:** Qwen3-TTS, VoxCPM2, Fish Audio, and Higgs Audio all served with minimal configuration changes.

## 1. Introduction: Why TTS in vLLM-Omni?

Text-to-speech has become a critical output modality for modern AI systems. Voice assistants, audiobook generation, real-time translation, and conversational agents all demand low-latency, high-quality speech synthesis at scale. Yet most serving engines were designed for text-in, text-out workloads — they lack the pipeline abstractions, streaming support, and multi-stage scheduling that TTS models require.

vLLM-Omni was built to solve exactly this class of problem. Its stage-based architecture decomposes complex generation pipelines into independently schedulable units, making it a natural fit for TTS models that chain text encoding, speech decoding, and vocoding into a single serving path. This post walks through how four diverse TTS models are served in vLLM-Omni, starting with deep architectural dives on Qwen3-TTS and VoxCPM2, followed by quick-start guides for Fish Audio and Higgs Audio.
```

- [ ] **Step 2: Verify the post renders locally (optional)**

```bash
bundle exec jekyll server
```

Open `http://localhost:4000` and confirm the post appears with correct title, author, and tags. Skip if Jekyll is not installed locally.

- [ ] **Step 3: Commit**

```bash
git add _posts/2026-06-10-serving-tts-vllm-omni.md
git commit -m "feat(blog): add TTS serving post — front matter and intro"
```

---

### Task 3: Write Section 2 — Deep Dive: Qwen3-TTS

**Files:**
- Modify: `_posts/2026-06-10-serving-tts-vllm-omni.md` (append after Section 1)

This section requires detailed technical content about vllm-omni's internals. The author should provide the actual architecture details. The structure below defines the exact sections and the writing prompts for each.

- [ ] **Step 1: Append the Qwen3-TTS deep dive section**

Append the following to `_posts/2026-06-10-serving-tts-vllm-omni.md`. Fill in each subsection with accurate technical details from the vllm-omni codebase and Qwen3-TTS model architecture.

```markdown
## 2. Deep Dive: Qwen3-TTS

### 2.1 Model Overview

[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) is a text-to-speech model from the Qwen family that generates high-quality speech from text input. It follows an autoregressive (AR) architecture, generating speech tokens sequentially before converting them to waveform output.

<!-- WRITING PROMPT: Describe Qwen3-TTS's architecture — text encoder, speech token decoder, vocoder chain. What are the input/output modalities? What makes it different from traditional TTS models? Include model size and key specs. -->

### 2.2 Stage Abstraction

vLLM-Omni decomposes Qwen3-TTS into a multi-stage pipeline, where each stage is an independently schedulable computation unit.

<!-- WRITING PROMPT: Describe the exact stage decomposition for Qwen3-TTS in vllm-omni. What are the stages (e.g., text encoding → token generation → vocoder)? How are stages defined in the codebase? How does the stage graph look? Include a diagram if available. -->

<p align="center">
<img src="/assets/figures/2026-06-10-serving-tts-vllm-omni/qwen3-tts-stage-graph.png" width="80%">
<br>
<em>Figure 1: Stage decomposition of Qwen3-TTS in vLLM-Omni.</em>
</p>

### 2.3 Streaming Support

Real-time TTS serving requires streaming — the client should receive audio chunks as they are generated, not wait for the entire utterance to complete. vLLM-Omni supports chunk-level audio streaming through its stage pipeline.

<!-- WRITING PROMPT: Explain how streaming works for Qwen3-TTS. How does the chunk boundary work? What is the client API for consuming streamed audio? Show the streaming flow from client request through API → stage pipeline → chunked response. Include latency numbers if available. -->

### 2.4 Async Chunk Scheduling

Async chunk scheduling (`async_chunk`) overlaps computation across stages and across requests, hiding latency and improving GPU utilization.

<!-- WRITING PROMPT: Explain async_chunk for Qwen3-TTS. How does it schedule chunks across stages? What parallelism opportunities does the AR architecture expose? How does it overlap computation? Include a timeline diagram if possible. -->

### 2.5 CUDA Graphs

CUDA graph capture eliminates kernel launch overhead by recording and replaying GPU execution graphs. For TTS models, this is particularly impactful in stages with repetitive computation patterns.

<!-- WRITING PROMPT: Describe how CUDA graphs are applied to Qwen3-TTS stages. Which stages benefit most? What is the capture strategy (static shapes? padding?)? What are warmup considerations? Quantify performance gains if benchmarks are available. -->

### 2.6 Supported Modes

Qwen3-TTS in vLLM-Omni supports multiple serving modes to cover different use cases.

<!-- WRITING PROMPT: List the serving modes — e.g., single-speaker, multi-speaker, voice clone, streaming vs. batch. How do they map to different stage configurations or parameters? -->

### 2.7 Quick Start

Serve Qwen3-TTS with a single command:

```bash
vllm serve Qwen/Qwen3-TTS --omni --port 8091
```

Call the API with streaming:

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

<!-- WRITING PROMPT: Replace the above commands with the actual serving command and API format for Qwen3-TTS in vllm-omni. Verify against the codebase/docs. -->
```

- [ ] **Step 2: Commit**

```bash
git add _posts/2026-06-10-serving-tts-vllm-omni.md
git commit -m "feat(blog): add Qwen3-TTS deep dive section"
```

---

### Task 4: Write Section 3 — Deep Dive: VoxCPM2

**Files:**
- Modify: `_posts/2026-06-10-serving-tts-vllm-omni.md` (append after Section 2)

- [ ] **Step 1: Append the VoxCPM2 deep dive section**

Append the following to `_posts/2026-06-10-serving-tts-vllm-omni.md`. This section mirrors Section 2's structure but focuses on VoxCPM2's diffusion-based architecture, highlighting how vllm-omni's abstractions handle a fundamentally different model type.

```markdown
## 3. Deep Dive: VoxCPM2

### 3.1 Model Overview

VoxCPM2 takes a different architectural approach from Qwen3-TTS. Rather than autoregressive token generation, it uses a diffusion-based pipeline to synthesize speech, offering distinct quality and latency trade-offs.

<!-- WRITING PROMPT: Describe VoxCPM2's architecture. How does diffusion-based TTS differ from AR TTS? What are the input modalities? Model size? Key architectural differences from Qwen3-TTS. -->

### 3.2 Stage Abstraction

The diffusion-based architecture creates a different stage decomposition pattern in vLLM-Omni compared to the AR pipeline in Qwen3-TTS.

<!-- WRITING PROMPT: Describe VoxCPM2's stage decomposition. Contrast with Qwen3-TTS's stage graph. How does the multi-step denoising loop map to stages? -->

<p align="center">
<img src="/assets/figures/2026-06-10-serving-tts-vllm-omni/voxcpm2-stage-graph.png" width="80%">
<br>
<em>Figure 2: Stage decomposition of VoxCPM2 in vLLM-Omni.</em>
</p>

### 3.3 Streaming Support

Streaming for diffusion-based TTS requires a different chunking strategy than AR models, since output tokens are generated through iterative refinement rather than sequential decoding.

<!-- WRITING PROMPT: Explain how streaming works for VoxCPM2. How is audio chunked during the diffusion process? What is the latency-to-first-chunk? How does it compare to Qwen3-TTS's streaming? -->

### 3.4 Async Chunk Scheduling

Diffusion models present different parallelism opportunities than AR models. The iterative denoising steps create scheduling patterns that async_chunk can exploit differently.

<!-- WRITING PROMPT: Explain async_chunk for VoxCPM2. How does scheduling differ from the AR case? What parallelism is available across denoising steps or across requests? -->

### 3.5 CUDA Graphs

CUDA graph capture for diffusion stages has different characteristics than for AR stages, particularly around warmup and shape stability across denoising steps.

<!-- WRITING PROMPT: Describe CUDA graphs for VoxCPM2. How do capture strategies differ from Qwen3-TTS? What are the warmup/replay characteristics for diffusion stages? -->

### 3.6 Supported Modes

<!-- WRITING PROMPT: List VoxCPM2's serving modes and how they differ from Qwen3-TTS's. -->

### 3.7 Quick Start

```bash
vllm serve openbmb/VoxCPM2 --omni --port 8091
```

```bash
curl -s http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openbmb/VoxCPM2",
    "input": "Welcome to VoxCPM2 TTS serving with vLLM-Omni.",
    "stream": true
  }'
```

<!-- WRITING PROMPT: Replace with actual serving command and API format for VoxCPM2. Verify model path and API parameters against the codebase. -->
```

- [ ] **Step 2: Commit**

```bash
git add _posts/2026-06-10-serving-tts-vllm-omni.md
git commit -m "feat(blog): add VoxCPM2 deep dive section"
```

---

### Task 5: Write Section 4 — Survey: Fish Audio & Higgs Audio

**Files:**
- Modify: `_posts/2026-06-10-serving-tts-vllm-omni.md` (append after Section 3)

- [ ] **Step 1: Append the survey section**

```markdown
## 4. Quick Start: Fish Audio & Higgs Audio

Beyond Qwen3-TTS and VoxCPM2, vLLM-Omni also supports Fish Audio and Higgs Audio, covering voice cloning and music/audio generation use cases.

### 4.1 Fish Audio

[Fish Audio](https://fish.audio/) specializes in voice cloning and custom voice synthesis. Given a short reference audio clip, it can generate speech in the target voice — useful for personalized TTS, character voices, and content creation.

<!-- WRITING PROMPT: 2-3 sentences on Fish Audio's capabilities. What makes it notable? Voice cloning quality, supported languages, etc. -->

```bash
vllm serve fishaudio/fish-speech-1.5 --omni --port 8091
```

<!-- WRITING PROMPT: Replace with actual model path and serving command. Add a curl example if the API differs from the standard speech endpoint. -->

### 4.2 Higgs Audio

[Higgs Audio](https://huggingface.co/Higgs-Labs/) extends TTS into music and general audio generation, enabling applications like background music synthesis and sound effect creation.

<!-- WRITING PROMPT: 2-3 sentences on Higgs Audio's capabilities. Music generation, audio quality, etc. -->

```bash
vllm serve Higgs-Labs/higgs-audio --omni --port 8091
```

<!-- WRITING PROMPT: Replace with actual model path and serving command. -->
```

- [ ] **Step 2: Commit**

```bash
git add _posts/2026-06-10-serving-tts-vllm-omni.md
git commit -m "feat(blog): add Fish Audio & Higgs Audio survey section"
```

---

### Task 6: Write Sections 5 & 6 — Comparison Table and Conclusion

**Files:**
- Modify: `_posts/2026-06-10-serving-tts-vllm-omni.md` (append after Section 4)

- [ ] **Step 1: Append the comparison table and conclusion**

```markdown
## 5. Model Comparison

| | **Qwen3-TTS** | **VoxCPM2** | **Fish Audio** | **Higgs Audio** |
|---|---|---|---|---|
| **Architecture** | Autoregressive | Diffusion | <!-- fill --> | <!-- fill --> |
| **Streaming** | ✅ | ✅ | <!-- fill --> | <!-- fill --> |
| **Voice cloning** | <!-- fill --> | <!-- fill --> | ✅ | <!-- fill --> |
| **Multilingual** | <!-- fill --> | <!-- fill --> | <!-- fill --> | <!-- fill --> |
| **VRAM** | <!-- fill --> | <!-- fill --> | <!-- fill --> | <!-- fill --> |
| **Use case** | General TTS | High-quality TTS | Voice cloning | Music / audio |

<!-- WRITING PROMPT: Fill in all cells with accurate data from benchmarks and model documentation. Add or remove rows as needed. -->

## 6. Conclusion & Getting Started

vLLM-Omni brings production-grade serving infrastructure to TTS models, with stage-based pipeline decomposition, real-time streaming, async chunk scheduling, and CUDA graph optimizations that work across fundamentally different architectures — from autoregressive models like Qwen3-TTS to diffusion-based models like VoxCPM2.

### Getting Started

```bash
pip install vllm-omni
vllm serve Qwen/Qwen3-TTS --omni --port 8091
```

### Resources

- **vLLM-Omni repository:** [github.com/vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)
- **Documentation:** [vllm-omni.readthedocs.io](https://vllm-omni.readthedocs.io) <!-- WRITING PROMPT: Update with actual docs URL -->
- **Model cards:** Qwen3-TTS · VoxCPM2 · Fish Audio · Higgs Audio <!-- WRITING PROMPT: Add actual HuggingFace links -->
- **Community:** [Discord](https://discord.gg/vllm) · [GitHub Discussions](https://github.com/vllm-project/vllm-omni/discussions)

We'd love to hear your feedback! Try serving these TTS models, open issues for bugs or feature requests, and contribute back to the vLLM-Omni ecosystem.

## Acknowledgements

<!-- WRITING PROMPT: List specific GitHub IDs of contributors to the TTS serving feature in vllm-omni. Format: "Special thanks to @github_handle1, @github_handle2, ... for their contributions to TTS model support in vLLM-Omni." -->
```

- [ ] **Step 2: Commit**

```bash
git add _posts/2026-06-10-serving-tts-vllm-omni.md
git commit -m "feat(blog): add comparison table and conclusion"
```

---

### Task 7: Add figures

**Files:**
- Add: `assets/figures/2026-06-10-serving-tts-vllm-omni/hero.png` (hero image)
- Add: `assets/figures/2026-06-10-serving-tts-vllm-omni/qwen3-tts-stage-graph.png` (Figure 1)
- Add: `assets/figures/2026-06-10-serving-tts-vllm-omni/voxcpm2-stage-graph.png` (Figure 2)
- Add: any additional figures referenced in the post

- [ ] **Step 1: Add all figures to the assets directory**

Copy or create the following images:
- `hero.png` — hero image for the post (used in front matter `image` field and social preview)
- `qwen3-tts-stage-graph.png` — stage decomposition diagram for Qwen3-TTS
- `voxcpm2-stage-graph.png` — stage decomposition diagram for VoxCPM2
- Any additional figures (async_chunk timelines, CUDA graph diagrams, comparison charts)

Images should be in PNG format, reasonable resolution for web (1200-1600px wide recommended).

- [ ] **Step 2: Commit**

```bash
git add assets/figures/2026-06-10-serving-tts-vllm-omni/
git commit -m "feat(blog): add figures for TTS serving post"
```

---

### Task 8: Fill in writing prompts and technical content

**Files:**
- Modify: `_posts/2026-06-10-serving-tts-vllm-omni.md`

This is the main content task. All `<!-- WRITING PROMPT: ... -->` comments in the post must be replaced with actual technical content, and all `<!-- fill -->` placeholders in the comparison table must be filled with accurate data.

- [ ] **Step 1: Fill in all writing prompts in Section 2 (Qwen3-TTS)**

Replace each `<!-- WRITING PROMPT: ... -->` comment with accurate technical content. Key sources:
- vllm-omni source code for stage definitions, streaming implementation, async_chunk, CUDA graph logic
- Qwen3-TTS model documentation on HuggingFace
- Internal benchmarks for performance numbers

Verify all code examples actually work against the current vllm-omni API.

- [ ] **Step 2: Fill in all writing prompts in Section 3 (VoxCPM2)**

Same process as Step 1, for VoxCPM2.

- [ ] **Step 3: Fill in all writing prompts in Sections 4, 5, and 6**

- Section 4: Fish Audio & Higgs Audio descriptions and serving commands
- Section 5: Comparison table — fill all `<!-- fill -->` cells with accurate data
- Section 6: Acknowledgements — list actual GitHub IDs; update docs URL and model card links

- [ ] **Step 4: Remove all remaining `<!-- WRITING PROMPT -->` and `<!-- fill -->` markers**

Search the file for any remaining placeholder markers and ensure none are left:

```bash
grep -n "WRITING PROMPT\|<!-- fill -->" _posts/2026-06-10-serving-tts-vllm-omni.md
```

Expected: no output (all placeholders resolved).

- [ ] **Step 5: Commit**

```bash
git add _posts/2026-06-10-serving-tts-vllm-omni.md
git commit -m "feat(blog): complete TTS serving post with technical content"
```

---

### Task 9: Local build verification

**Files:**
- Verify: `_posts/2026-06-10-serving-tts-vllm-omni.md`
- Verify: all figure references

- [ ] **Step 1: Build the site locally and verify**

```bash
bundle exec jekyll build
```

Check for build errors. Common issues:
- Broken image paths (figure `src` must match actual files in `assets/figures/`)
- YAML front matter syntax errors
- Unclosed HTML tags in figure blocks

- [ ] **Step 2: Preview the post**

```bash
bundle exec jekyll server
```

Open `http://localhost:4000` and verify:
- Post appears on homepage with correct title and excerpt
- Post renders with correct formatting (headers, code blocks, figures, table)
- All images load correctly
- Tags display correctly in post footer

- [ ] **Step 3: Commit any fixes**

If the build or preview reveals issues, fix them and commit:

```bash
git add -A
git commit -m "fix(blog): fix build issues in TTS serving post"
```

---

### Task 10: Final commit and push

- [ ] **Step 1: Review the full diff**

```bash
git log --oneline main..HEAD
git diff main..HEAD --stat
```

Review all commits on the branch. Verify:
- Only `_posts/2026-06-10-serving-tts-vllm-omni.md` and `assets/figures/2026-06-10-serving-tts-vllm-omni/` are changed
- No leftover writing prompt comments or `<!-- fill -->` markers
- Figure paths in the post match actual files
- All code examples use correct vllm-omni API syntax

- [ ] **Step 2: Push the branch**

```bash
git push -u origin blog/serving-tts-vllm-omni
```
