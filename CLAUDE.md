# dLLM - Diffusion Language Modeling

This document provides context for Claude Code when working with the dLLM codebase.

## Overview

**dLLM** is a library for training language models using **Masked Diffusion Language Models (MDLM)**, as described in [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524). Unlike traditional autoregressive models, MDLM learns to denoise progressively masked tokens through a diffusion process.

Key features:
- Masked diffusion training with timestep sampling
- Support for text and multi-modal (vision-language) models
- LoRA and full fine-tuning support
- Distributed training with FSDP and DeepSpeed
- Easy integration with HuggingFace models

## Repository Structure

```
dllm/
├── dllm/
│   ├── core/               # Core MDLM implementation
│   │   ├── models/         # Base masked LM models
│   │   └── trainers/       # MDLMTrainer (diffusion training logic)
│   ├── data/               # Dataset loaders and utilities
│   │   ├── alpaca.py       # Alpaca instruction dataset
│   │   ├── ko_document_table.py  # Korean document/table VQA
│   │   └── utils.py        # Dataset loading dispatcher
│   ├── pipelines/          # Model-specific implementations
│   │   ├── llama3/         # LLaMA 3 with MDLM
│   │   ├── gemma2/         # Gemma 2 with MDLM
│   │   └── qwen3_vl/       # Qwen3-VL with MDLM (vision-language)
│   │       ├── models/     # Qwen3VLForMaskedLM
│   │       ├── trainer.py  # Qwen3VLTrainer
│   │       └── utils.py    # Data collator for multi-modal
│   └── utils/              # Training utilities
├── examples/               # Training scripts
│   ├── llama3/
│   ├── gemma2/
│   └── qwen3_vl/           # Qwen3-VL examples
│       ├── sft.py          # Supervised fine-tuning script
│       └── README.md       # Usage documentation
├── scripts/                # Helper scripts
│   ├── accelerate_configs/ # Accelerate config files (DDP, FSDP, DeepSpeed)
│   ├── train.slurm.sh      # Slurm training script
│   └── test_qwen3_vl_setup.py  # Test Qwen3-VL integration
└── docs/                   # Documentation
```

## Quick Start Commands

### Single GPU Training
```bash
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/llama3/sft.py \
    --lora True \
    --per_device_train_batch_size 4
```

### Multi-GPU Training (8 GPUs)
```bash
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/llama3/sft.py \
    --lora True
```

### Slurm Cluster
```bash
sbatch --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/llama3/sft.py" \
    --lora True
```

### Test Installation
```bash
# Test Qwen3-VL setup
python scripts/test_qwen3_vl_setup.py

# Run with specific model
python examples/llama3/sft.py --help
```

## Core Concepts

### 1. Masked Diffusion Language Models (MDLM)

MDLM is a non-autoregressive training approach that:
1. **Samples timestep** t ~ Uniform(0, 1)
2. **Masks tokens** stochastically according to masking schedule α(t)
3. **Predicts masked tokens** using the language model
4. **Computes weighted loss** based on timestep

**Key equation**: At timestep t, each token has probability α(t) of being masked.

**Training objective**:
```
L = E[w(t) * CrossEntropy(predictions, targets)]
```
where w(t) is a timestep-dependent weight.

**Advantages over autoregressive**:
- Can learn bidirectional context
- More flexible generation (can fill in any position)
- Often faster inference via iterative denoising

### 2. MDLMTrainer

The core trainer is in `dllm/core/trainers/mdlm_trainer.py`:

```python
class MDLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. Sample timesteps
        # 2. Apply masking
        # 3. Forward pass
        # 4. Compute weighted loss
```

All model-specific trainers should extend `MDLMTrainer`:
- `Qwen3VLTrainer` extends it for multi-modal inputs
- Future trainers can override `_preprocess_inputs()` for custom preprocessing

### 3. Model Wrappers

Each model has a wrapper that:
- Wraps the base HuggingFace model
- Returns `MaskedLMOutput` (required for MDLM)
- Provides model-specific functionality (e.g., freezing vision components)

Example: `Qwen3VLForMaskedLM` wraps `Qwen3VLForConditionalGeneration`

```python
from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM

model = Qwen3VLForMaskedLM.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    freeze_vision=True,    # Freeze vision encoder
    freeze_merger=True,    # Freeze projection layer
)
```

### 4. Data Format

All datasets should use **OpenAI-compatible message format**:

```python
{
    "messages": [
        {"role": "user", "content": "What is in this image?"},
        {"role": "assistant", "content": "This image shows..."}
    ],
    "image": <PIL.Image>  # Optional, for multi-modal datasets
}
```

Multi-turn conversations are supported:
```python
{
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "I don't have weather data."}
    ]
}
```

## Architecture Deep Dive

### Text-Only Models (LLaMA 3, Gemma 2)

```
Input → Tokenizer → Model → MaskedLMOutput
                      ↓
                [MASK] tokens at timestep t
                      ↓
                  LM predicts
                      ↓
                 Weighted loss
```

### Multi-Modal Models (Qwen3-VL)

```
Image → Vision Encoder → Merger → Combined
Text  → Tokenizer     ↗          Embeddings
                                     ↓
                                   Model
                                     ↓
                              MaskedLMOutput
```

**Vision component freezing**:
- Vision Encoder: Extracts visual features from images
- Merger: Projects visual features to language model dimension
- Language Model: Processes combined visual + text embeddings

In fine-tuning, we typically:
1. **Freeze** vision encoder and merger (pre-trained features are good)
2. **Train** language model with LoRA (parameter-efficient)
3. **Focus** on language understanding of visual content

## Qwen3-VL Integration

### Files Created

1. **Data Loader**: `dllm/data/ko_document_table.py`
   - Loads jp1924/KoDocumentTableVisualSFT dataset
   - Converts JSON content to OpenAI message format

2. **Model**: `dllm/pipelines/qwen3_vl/models/modeling_qwen3_vl.py`
   - `Qwen3VLForMaskedLM` wrapper
   - `freeze_vision_components()` method
   - Returns `MaskedLMOutput`

3. **Data Collator**: `dllm/pipelines/qwen3_vl/utils.py`
   - `Qwen3VLDataCollator` for multi-modal batches
   - Converts messages to Qwen3VL format
   - Handles prompt loss masking

4. **Trainer**: `dllm/pipelines/qwen3_vl/trainer.py`
   - `Qwen3VLTrainer` extends `MDLMTrainer`
   - Preprocesses vision tensors (device placement)

5. **Training Script**: `examples/qwen3_vl/sft.py`
   - Complete training pipeline
   - LoRA support
   - Freezes vision components

6. **Test Script**: `scripts/test_qwen3_vl_setup.py`
   - Validates entire integration
   - Tests data loading, model, collator, forward pass

### Usage Example

```python
from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM
from dllm.pipelines.qwen3_vl.trainer import Qwen3VLTrainer
from dllm.pipelines.qwen3_vl.utils import create_qwen3_vl_collator

# 1. Load model with frozen vision
model = Qwen3VLForMaskedLM.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    freeze_vision=True,
    freeze_merger=True,
)

# 2. Apply LoRA
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)

# 3. Load processor
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# 4. Create collator
collator = create_qwen3_vl_collator(processor, mask_prompt_loss=True)

# 5. Train
trainer = Qwen3VLTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)
trainer.train()
```

## Common Tasks

### Adding a New Model

1. Create model wrapper in `dllm/pipelines/<model_name>/models/`
2. Extend `MDLMTrainer` if needed in `dllm/pipelines/<model_name>/trainer.py`
3. Create data collator if needed
4. Add training script in `examples/<model_name>/sft.py`
5. Update `dllm/pipelines/<model_name>/__init__.py`

### Adding a New Dataset

1. Create loader in `dllm/data/<dataset_name>.py`
2. Register in `dllm/data/utils.py:load_sft_dataset()`
3. Ensure output format: `{"messages": [...], "image": ...}`

### Debugging Training

```bash
# Check GPU usage
nvidia-smi

# Monitor training
tail -f <output_dir>/training.log

# Test on small data
python examples/qwen3_vl/sft.py \
    --dataset_args "jp1924/KoDocumentTableVisualSFT[train:100,test:10]" \
    --num_train_epochs 1

# Test forward pass
python scripts/test_qwen3_vl_setup.py
```

### Memory Optimization

1. **Use LoRA**: `--lora True --r 16`
2. **Reduce batch size**: `--per_device_train_batch_size 2`
3. **Gradient accumulation**: `--gradient_accumulation_steps 8`
4. **Gradient checkpointing**: `--gradient_checkpointing True`
5. **Mixed precision**: `--bf16 True` (recommended for A100/H100)
6. **Freeze components**: For multi-modal, freeze vision encoder

### Distributed Training

**FSDP** (Fully Sharded Data Parallel):
```bash
accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/qwen3_vl/sft.py --lora True
```

**DeepSpeed**:
```bash
accelerate launch --config_file scripts/accelerate_configs/deepspeed.yaml \
    examples/qwen3_vl/sft.py --lora True
```

**Multi-node FSDP** (Slurm):
```bash
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/qwen3_vl/sft.py"
```

## Key Implementation Details

### Masking Schedule

The masking schedule α(t) determines how many tokens are masked at timestep t:
- t=0: No masking (fully observed)
- t=1: Maximum masking
- Common schedules: linear, cosine, square root

Implemented in `dllm/core/trainers/mdlm_trainer.py`

### Loss Weighting

Loss is weighted by timestep to balance learning:
- Early timesteps (few masks): Lower weight
- Late timesteps (many masks): Higher weight

This prevents the model from focusing too much on easy examples.

### Prompt Loss Masking

For instruction tuning, we mask the prompt (user messages) in loss computation:
```python
collator = create_qwen3_vl_collator(
    processor=processor,
    mask_prompt_loss=True,  # Only compute loss on assistant responses
)
```

This is implemented in `Qwen3VLDataCollator._mask_prompt_tokens()`.

### Vision Component Freezing

In `Qwen3VLForMaskedLM.freeze_vision_components()`:
```python
# Freeze visual encoder
for param in self.model.visual.parameters():
    param.requires_grad = False

# Freeze merger (projection)
for param in self.model.merger.parameters():
    param.requires_grad = False
```

Benefits:
- Faster training (fewer parameters to update)
- Lower memory usage
- Stable visual features
- Focus on language understanding

### Device Placement for Multi-Modal

Vision tensors need explicit device placement:
```python
def _preprocess_inputs(self, inputs):
    vision_keys = ["pixel_values", "image_grid_thw"]
    for key in vision_keys:
        if key in inputs:
            inputs[key] = inputs[key].to(self.args.device)
```

This is handled automatically in `Qwen3VLTrainer`.

## Troubleshooting

### Import Errors

**Problem**: `ImportError: cannot import name 'Qwen3VLForMaskedLM'`

**Solution**:
```bash
# Reinstall in development mode
pip install -e .

# Check installation
python -c "from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM; print('OK')"
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: `--per_device_train_batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing True`
3. Use LoRA instead of full fine-tuning
4. Reduce sequence length: `--max_seq_length 1024`
5. Use FSDP for distributed training

### Slow Training

**Problem**: Training is very slow

**Checks**:
1. Is `bf16=True`? (much faster than fp32)
2. Is `flash_attention_2` being used?
3. Are vision components frozen? (for Qwen3-VL)
4. Is `dataloader_num_workers > 0`?

### Poor Results

**Problem**: Model doesn't learn well

**Checks**:
1. Is `mask_prompt_loss=True`? (for instruction tuning)
2. Is learning rate appropriate? (try 1e-4 to 5e-4)
3. Are enough epochs being trained? (try 3-5)
4. Is the dataset format correct?

### Vision Model Issues

**Problem**: Qwen3-VL forward pass fails

**Checks**:
1. Are images in PIL format?
2. Is `pixel_values` on correct device?
3. Is `image_grid_thw` present in inputs?
4. Is processor from same checkpoint as model?

## Development Workflow

### Testing Changes

```bash
# 1. Run unit tests (if available)
pytest tests/

# 2. Test on small dataset
python scripts/test_qwen3_vl_setup.py

# 3. Quick training test (1 GPU, small data)
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml \
    --num_processes 1 examples/qwen3_vl/sft.py \
    --dataset_args "jp1924/KoDocumentTableVisualSFT[train:10,test:2]" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1

# 4. Full training
accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/qwen3_vl/sft.py
```

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public functions/classes
- Keep functions focused (single responsibility)

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes and commit
git add .
git commit -m "Add new model integration"

# Push to fork
git push origin feature/new-model

# Create PR on GitHub
```

## Resources

### Papers
- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [Qwen3-VL: To See the World More Clearly](https://qwenlm.github.io/blog/qwen3-vl/)

### Documentation
- HuggingFace Transformers: https://huggingface.co/docs/transformers
- Accelerate: https://huggingface.co/docs/accelerate
- PEFT (LoRA): https://huggingface.co/docs/peft

### Community
- GitHub Issues: https://github.com/yourusername/dllm/issues
- Discussions: https://github.com/yourusername/dllm/discussions

## License

See [LICENSE](LICENSE) file.

---

**Last Updated**: 2025-01-10

For questions or issues, please open a GitHub issue or discussion.
