# Qwen3-VL with Diffusion Language Modeling

This directory contains examples for fine-tuning **Qwen3-VL** models using **Masked Diffusion Language Modeling (MDLM)** on multi-modal datasets.

## Overview

This integration combines:
- **Qwen3-VL-2B-Instruct**: A vision-language model with strong multi-modal capabilities
- **MDLM**: Masked diffusion training that learns to denoise progressively masked tokens
- **LoRA**: Parameter-efficient fine-tuning for the language model
- **Frozen Vision Components**: Only the language model is trained; vision encoder and projection are frozen

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers accelerate peft torch datasets pillow qwen-vl-utils
```

### 2. Single GPU Training (for testing)

```bash
accelerate launch \
    --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/qwen3_vl/sft.py \
    --lora True \
    --per_device_train_batch_size 2
```

### 3. Multi-GPU Training (8 GPUs with FSDP)

```bash
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp.yaml \
    examples/qwen3_vl/sft.py \
    --lora True \
    --per_device_train_batch_size 4
```

### 4. Slurm Cluster Training

**Single node (8 GPUs):**
```bash
sbatch --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/qwen3_vl/sft.py" \
    --lora True
```

**Multi-node (2 nodes, 16 GPUs):**
```bash
sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/qwen3_vl/sft.py" \
    --lora True
```

## Key Arguments

### Model Arguments

- `--model_name_or_path`: Path or name of the Qwen3-VL model (default: `Qwen/Qwen3-VL-2B-Instruct`)
- `--freeze_vision`: Freeze vision encoder (default: `True`)
- `--freeze_merger`: Freeze visual projection/merger layer (default: `True`)
- `--lora`: Enable LoRA for parameter-efficient training (default: `False`)
- `--r`: LoRA rank (default: `16`)
- `--lora_alpha`: LoRA alpha scaling (default: `32`)
- `--lora_dropout`: LoRA dropout rate (default: `0.05`)
- `--target_modules`: Comma-separated list of modules to apply LoRA (default: `"q_proj,k_proj,v_proj,o_proj"`)

### Data Arguments

- `--dataset_args`: Dataset name or path (default: `jp1924/KoDocumentTableVisualSFT`)
  - Use `[train:N,test:M]` to limit dataset size (e.g., `jp1924/KoDocumentTableVisualSFT[train:1000,test:100]`)
- `--mask_prompt_loss`: Mask prompt tokens in loss computation (default: `True`)
- `--max_seq_length`: Maximum sequence length (default: `2048`)

### Training Arguments

- `--output_dir`: Output directory for checkpoints (default: `outputs/qwen3-vl-2b-diffusion-ko-document`)
- `--num_train_epochs`: Number of training epochs (default: `3`)
- `--per_device_train_batch_size`: Batch size per GPU (default: `4`)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: `2`)
- `--learning_rate`: Learning rate (default: `2e-4`)
- `--warmup_ratio`: Warmup ratio (default: `0.03`)
- `--lr_scheduler_type`: LR scheduler type (default: `"cosine"`)
- `--bf16`: Use bfloat16 precision (default: `True`)
- `--tf32`: Use TF32 on Ampere GPUs (default: `True`)

## Dataset Format

The training script expects datasets in **OpenAI-compatible message format**:

```python
{
    "messages": [
        {"role": "user", "content": "Describe this image."},
        {"role": "assistant", "content": "This is a document containing..."}
    ],
    "image": <PIL.Image>
}
```

### Supported Datasets

- **jp1924/KoDocumentTableVisualSFT**: Korean document/table QA dataset
  - Automatically converts JSON-formatted content to message format
  - Contains images of documents, tables, and charts with Korean text

### Custom Datasets

To use your own dataset, ensure it has the following structure:
1. A `messages` column with conversation history
2. An `image` column with PIL Images
3. Each message should have `role` and `content` fields

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Qwen3VLForMaskedLM                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────┐      ┌──────────────┐              │
│  │ Vision Encoder│      │   Merger     │              │
│  │   (Frozen)    │─────▶│  (Frozen)    │              │
│  └───────────────┘      └──────┬───────┘              │
│                                 │                       │
│                         ┌───────▼───────────┐          │
│                         │ Language Model    │          │
│                         │ (Trainable + LoRA)│          │
│                         └───────────────────┘          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Training Process

1. **Image Processing**: Images are processed by the frozen vision encoder
2. **Feature Projection**: Visual features are projected to language space (frozen)
3. **Text Tokenization**: Text is tokenized and combined with visual tokens
4. **Diffusion Masking**: Tokens are stochastically masked according to timestep t
5. **Forward Pass**: The language model (with LoRA) predicts masked tokens
6. **Loss Computation**: Weighted cross-entropy loss on masked positions
7. **Backpropagation**: Only LoRA parameters are updated

### Why Freeze Vision Components?

- **Faster Training**: Reduces GPU memory and computation
- **Stable Features**: Pre-trained vision encoder already extracts good features
- **Focus on Language**: Diffusion training focuses on the language modeling task
- **Parameter Efficiency**: Combined with LoRA for minimal trainable parameters

## Memory Requirements

Approximate GPU memory usage (with LoRA, batch_size=1):

| Model Size | BF16 | FP32 |
|------------|------|------|
| Qwen3-VL-2B | ~8GB | ~16GB |
| Qwen3-VL-7B | ~18GB | ~36GB |

For multi-GPU training with FSDP, memory is distributed across GPUs.

## Example: Custom Training

```python
from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM
from dllm.pipelines.qwen3_vl.trainer import Qwen3VLTrainer
from dllm.pipelines.qwen3_vl.utils import create_qwen3_vl_collator
from transformers import AutoProcessor, TrainingArguments

# Load model with frozen vision components
model = Qwen3VLForMaskedLM.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    freeze_vision=True,
    freeze_merger=True,
)

# Apply LoRA
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# Create data collator
collator = create_qwen3_vl_collator(
    processor=processor,
    mask_prompt_loss=True,
    max_seq_length=2048,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    bf16=True,
)

# Create trainer
trainer = Qwen3VLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

# Train
trainer.train()
```

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Enable gradient checkpointing (add `--gradient_checkpointing True`)

### Slow Training

- Ensure `bf16=True` and `tf32=True`
- Use `flash_attention_2` (requires `flash-attn` package)
- Check `dataloader_num_workers` (default: 4)

### Poor Results

- Increase training epochs
- Adjust learning rate (try 1e-4 to 5e-4)
- Try different LoRA ranks (8, 16, 32, 64)
- Ensure `mask_prompt_loss=True` for instruction tuning

## Citation

If you use this code, please cite:

```bibtex
@article{sahoo2024simple,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Sahoo, Subham Sekhar and Oymak, Samet and Palangi, Hamid and Celikyilmaz, Asli},
  journal={arXiv preprint arXiv:2406.07524},
  year={2024}
}

@article{qwen3vl,
  title={Qwen3-VL: To See the World More Clearly},
  author={Qwen Team},
  year={2024}
}
```

## License

See the main repository [LICENSE](../../LICENSE) file.
