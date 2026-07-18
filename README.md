# Hugging Face Hands-on: Multi-Modal Operations

This repository demonstrates how I implemented **multiple Hugging Face capabilities across text, audio, image, video, and UI workflows** using notebooks.  
The project combines `transformers`, `diffusers`, `datasets`, and `gradio` to build practical end-to-end inference flows.

## What is implemented

| Notebook | Hugging Face functionality | Implementation details |
|---|---|---|
| `transformer-hugging-face.ipynb` | Core NLP pipelines | Implemented `pipeline()`-based tasks for **text generation**, **sentiment analysis**, **financial sentiment (FinBERT)**, **NER**, **question answering**, and **English→French translation**. |
| `auto_processing.ipynb` | Audio understanding + speech generation | Implemented `AutoFeatureExtractor` + `ASTForAudioClassification` (`MIT/ast-finetuned-audioset-10-10-0.4593`) for audio feature/class tasks, `pipeline(\"automatic-speech-recognition\")` for speech-to-text, and `pipeline(\"text-to-speech\")` / SpeechT5 usage for text-to-audio. |
| `diffuser.ipynb` | Image generation with diffusion | Implemented `DDPMPipeline`, `DDPMScheduler`, and `UNet2DModel` with `google/ddpm-celebahq-256` for denoising-based generation, plus `DiffusionPipeline` with `stabilityai/stable-diffusion-xl-base-1.0` for prompt-driven image synthesis. |
| `video-model.ipynb` | Image-to-video generation | Implemented `StableVideoDiffusionPipeline` using `stabilityai/stable-video-diffusion-img2vid-xt` to generate video frames from an image + prompt and export to video output. |
| `gradio-hf.ipynb` | Interactive ML apps | Built Gradio interfaces from basic components (`Number`, `Textbox`, `Slider`, `Dropdown`, `Image`) to structured `Blocks`, tabs, accordion sections, custom buttons, and CSS; integrated model-backed examples like image classification and sentiment prediction. |

## Brief functionality breakdown (implementation-focused)

1. **Text generation and NLP analysis**  
   The transformer notebook loads pretrained language models and runs prompt-based generation, sentiment prediction, NER extraction, QA inference, and translation using the `pipeline` API plus tokenizer/model flows where needed.

2. **Domain-specific sentiment with FinBERT**  
   In addition to generic sentiment models, the implementation uses `ProsusAI/finbert` for financial-text sentiment so outputs are more aligned with finance vocabulary and context.

3. **Audio classification and feature extraction**  
   The audio notebook applies `AutoFeatureExtractor` with AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) to convert raw waveform input into model-ready features and infer audio classes.

4. **Speech-to-text (ASR)**  
   Audio files are passed through `pipeline("automatic-speech-recognition")` to transcribe spoken content into text outputs for downstream NLP processing.

5. **Text-to-speech (TTS)**  
   The same workflow also includes `pipeline("text-to-speech")` / SpeechT5-style generation to convert text back into playable audio, enabling round-trip speech workflows.

6. **Image synthesis via diffusion (DDPM + SDXL)**  
   The diffusion notebook demonstrates both low-level denoising setup (`DDPMScheduler`, `UNet2DModel`) and high-level prompt-to-image generation with SDXL through `DiffusionPipeline`.

7. **Video generation from image + prompt**  
   The video notebook uses `StableVideoDiffusionPipeline` to transform a seed image and text guidance into frame sequences and exports those frames as a final video artifact.

8. **Interactive app layer with Gradio**  
   The Gradio notebook wraps model logic inside web interfaces (components + `Blocks`) so each functionality can be run interactively without editing core notebook code.

9. **Cross-modal Hugging Face operation pattern**  
   Across all notebooks, the shared implementation pattern is: load pretrained artifact from Hub → preprocess modality input → run inference/generation → post-process output → expose as reusable UI or media artifact.

## Multi-functionality design across the repository

The implementation is intentionally split by modality and then connected by common Hugging Face operations:

1. **Pipeline-first inference pattern**: rapid task setup with `transformers.pipeline(...)` for NLP and speech operations.
2. **Auto-class abstractions**: model loading with `Auto*` classes (`AutoTokenizer`, `AutoFeatureExtractor`, etc.) for consistent model interchangeability.
3. **Diffusion specialization**: image and video generation handled with `diffusers` pipelines/schedulers for iterative denoising workflows.
4. **Interface layer**: Gradio notebooks convert notebook experiments into reusable interactive tools.
5. **Artifact flow**: repository artifacts (such as `audio.mp3`, generated audio/video outputs, and sample images) support round-trip testing between input media and generated results.

## Operational workflow used

1. **Load pretrained model or task pipeline** from Hugging Face Hub.
2. **Preprocess modality-specific input** (text/audio/image tensors, feature extraction, resizing/loading).
3. **Run inference/generation** using transformers or diffusion pipelines.
4. **Post-process outputs** (labels, generated text, synthesized audio, frame sequences, exported videos).
5. **Expose interaction in Gradio** where applicable for repeatable user-driven execution.

## Tech stack

- `transformers`
- `diffusers`
- `datasets`
- `huggingface-hub`
- `torch`, `torchaudio`, `torchvision`
- `gradio`
- `librosa`, `pydub`, `opencv-python`, `matplotlib`, `numpy`

## Setup

```bash
pip install -r requirement.txt
```

or (using project metadata):

```bash
pip install .
```

## Repository purpose

This codebase is a practical Hugging Face playground showing **how multiple model families and operations can be implemented together**: traditional transformer NLP, speech pipelines, diffusion image synthesis, diffusion video generation, and UI deployment patterns with Gradio.