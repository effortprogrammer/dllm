"""
Qwen3-VL Supervised Fine-Tuning with Diffusion Language Modeling

This script fine-tunes Qwen3-VL with masked diffusion on multi-modal data.
It freezes the vision encoder and projection, training only the language model with LoRA.

Local users
------------
- 1 GPU (LoRA, for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/qwen3_vl/sft.py \
        --lora True \
        --per_device_train_batch_size 2

- 8 GPUs (FSDP with LoRA):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/qwen3_vl/sft.py \
        --lora True

Slurm users
------------
- 1 Node, 8 GPUs (FSDP):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/qwen3_vl/sft.py" \
        --lora True

- 2 Nodes, 16 GPUs (FSDP):
    sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/qwen3_vl/sft.py" \
        --lora True
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
import accelerate

import dllm
from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM
from dllm.pipelines.qwen3_vl.trainer import Qwen3VLTrainer
from dllm.pipelines.qwen3_vl.utils import create_qwen3_vl_collator


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-2B-Instruct",
        metadata={"help": "Path to pretrained Qwen3-VL model"}
    )
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Whether to freeze vision encoder"}
    )
    freeze_merger: bool = field(
        default=True,
        metadata={"help": "Whether to freeze visual merger (projection layer)"}
    )
    attn_implementation: str = field(
        default="eager",
        metadata={"help": "Attention implementation to use. Options: eager, flash_attention_2, sdpa. Default: eager"}
    )
    # Override target_modules for Qwen3-VL specifically
    # Language model attention: q_proj, k_proj, v_proj, o_proj
    # Language model MLP: gate_proj, up_proj, down_proj
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA target modules for Qwen3-VL language model"}
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = field(
        default="jp1924/KoDocumentTableVisualSFT",
        metadata={"help": "Dataset name or path. Use [train:N,test:M] to limit size"}
    )
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for training (increased for image tokens)"}
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "outputs/qwen3-vl-2b-diffusion-ko-document"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    bf16: bool = True
    tf32: bool = True
    dataloader_num_workers: int = 4
    group_by_length: bool = False  # Don't group by length for multimodal
    report_to: str = "wandb"
    run_name: Optional[str] = None
    remove_unused_columns: bool = False  # Required for custom data collator


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    dllm.utils.print_main("Loading Qwen3-VL model...")

    # Load model with our wrapper
    model = Qwen3VLForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(model_args, "dtype", "bfloat16"),
        attn_implementation=model_args.attn_implementation,
        device_map={"": accelerate.PartialState().local_process_index}
        if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        else None,
        freeze_vision=model_args.freeze_vision,
        freeze_merger=model_args.freeze_merger,
    )

    # Apply LoRA if requested
    if getattr(model_args, "lora", False):
        dllm.utils.print_main("Applying LoRA to language model...")
        from peft import LoraConfig, get_peft_model

        # Parse target modules
        target_modules = model_args.target_modules.split(",")
        dllm.utils.print_main(f"LoRA target modules: {target_modules}")

        lora_config = LoraConfig(
            r=model_args.r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.bias,
            task_type="CAUSAL_LM",
            modules_to_save=model_args.modules_to_save.split(",")
            if model_args.modules_to_save else None,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ----- Processor (tokenizer + image processor) --------------------------------
    dllm.utils.print_main("Loading processor...")
    processor = transformers.AutoProcessor.from_pretrained(
        model_args.model_name_or_path
    )

    # Set processor padding side
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = "right"

    # Ensure pad_token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Add mask token if not present
    if not hasattr(processor.tokenizer, 'mask_token_id') or processor.tokenizer.mask_token_id is None:
        special_tokens_dict = {'mask_token': '<|mask|>'}
        num_added_toks = processor.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_toks > 0:
            model.resize_token_embeddings(len(processor.tokenizer))
            dllm.utils.print_main(f"Added {num_added_toks} special tokens (mask_token)")

    # ----- Dataset ----------------------------------------------------------------
    dllm.utils.print_main("Loading dataset...")
    with accelerate.PartialState().local_main_process_first():
        # Load multi-modal dataset
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )

        dllm.utils.print_main(f"Dataset loaded: train={len(dataset['train'])}, test={len(dataset.get('test', []))}")

        # Note: We don't apply tokenization here because Qwen3VLDataCollator
        # will handle both image and text processing together

    # ----- Data Collator ----------------------------------------------------------
    data_collator = create_qwen3_vl_collator(
        processor=processor,
        mask_prompt_loss=data_args.mask_prompt_loss,
        max_seq_length=data_args.max_seq_length,
    )

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    dllm.utils.print_main("Starting training...")

    trainer = Qwen3VLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        data_collator=data_collator,
        processing_class=processor,  # Add processor for MDLMTrainer
    )

    # Train
    trainer.train()

    # Save final checkpoint
    dllm.utils.print_main("Saving final checkpoint...")
    final_checkpoint_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_checkpoint_dir)
    processor.save_pretrained(final_checkpoint_dir)

    dllm.utils.print_main(f"Training complete! Model saved to {final_checkpoint_dir}")


if __name__ == "__main__":
    train()
