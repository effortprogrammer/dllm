#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 dLLM team. All rights reserved.
"""
Test script to verify Qwen3-VL integration with dLLM.

This script tests:
1. Data loading (KoDocumentTableVisualSFT)
2. Model initialization (Qwen3VLForMaskedLM)
3. Data collator (Qwen3VLDataCollator)
4. Forward pass with multi-modal inputs
5. Vision component freezing

Usage:
    python scripts/test_qwen3_vl_setup.py
"""

import sys
import torch
from transformers import AutoProcessor

print("=" * 80)
print("Qwen3-VL Integration Test")
print("=" * 80)

# Test 1: Import dllm modules
print("\n[1/6] Testing imports...")
try:
    import dllm
    from dllm.data import load_sft_dataset
    from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM
    from dllm.pipelines.qwen3_vl.trainer import Qwen3VLTrainer
    from dllm.pipelines.qwen3_vl.utils import create_qwen3_vl_collator
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load dataset (small subset for testing)
print("\n[2/6] Testing dataset loading...")
try:
    dataset = load_sft_dataset("jp1924/KoDocumentTableVisualSFT[train:10,test:2]")
    print(f"✓ Dataset loaded: train={len(dataset['train'])}, test={len(dataset['test'])}")

    # Check dataset structure
    sample = dataset['train'][0]
    assert 'messages' in sample, "Dataset missing 'messages' field"
    assert 'image' in sample, "Dataset missing 'image' field"
    print(f"✓ Dataset structure valid")
    print(f"  - Sample messages: {len(sample['messages'])} messages")
    print(f"  - Sample image: {sample['image'].size}")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    sys.exit(1)

# Test 3: Load model and processor
print("\n[3/6] Testing model and processor loading...")
try:
    model = Qwen3VLForMaskedLM.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Use CPU for testing
        freeze_vision=True,
        freeze_merger=True,
    )
    print("✓ Model loaded successfully")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    print("✓ Processor loaded successfully")

    # Ensure mask token is present
    if not hasattr(processor.tokenizer, 'mask_token_id') or processor.tokenizer.mask_token_id is None:
        special_tokens_dict = {'mask_token': '<|mask|>'}
        num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
        if num_added > 0:
            model.resize_token_embeddings(len(processor.tokenizer))
            print(f"✓ Added mask token to tokenizer")

except Exception as e:
    print(f"✗ Model/processor loading failed: {e}")
    sys.exit(1)

# Test 4: Check vision component freezing
print("\n[4/6] Testing vision component freezing...")
try:
    vision_frozen = True
    merger_frozen = True

    # Check visual encoder
    if hasattr(model.model, 'visual'):
        for param in model.model.visual.parameters():
            if param.requires_grad:
                vision_frozen = False
                break

    # Check merger
    if hasattr(model.model, 'merger'):
        for param in model.model.merger.parameters():
            if param.requires_grad:
                merger_frozen = False
                break

    assert vision_frozen, "Vision encoder not frozen!"
    assert merger_frozen, "Merger not frozen!"
    print("✓ Vision encoder frozen")
    print("✓ Merger frozen")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")

except Exception as e:
    print(f"✗ Freezing check failed: {e}")
    sys.exit(1)

# Test 5: Create data collator
print("\n[5/6] Testing data collator...")
try:
    collator = create_qwen3_vl_collator(
        processor=processor,
        mask_prompt_loss=True,
        max_seq_length=512,  # Shorter for testing
    )
    print("✓ Data collator created")

    # Test collator on a batch
    batch = [dataset['train'][0], dataset['train'][1]]
    batch_output = collator(batch)

    assert 'input_ids' in batch_output, "Missing input_ids"
    assert 'attention_mask' in batch_output, "Missing attention_mask"
    assert 'labels' in batch_output, "Missing labels"
    assert 'pixel_values' in batch_output, "Missing pixel_values"

    print(f"✓ Batch collation successful")
    print(f"  - input_ids shape: {batch_output['input_ids'].shape}")
    print(f"  - pixel_values shape: {batch_output['pixel_values'].shape}")
    print(f"  - labels shape: {batch_output['labels'].shape}")

    # Check that some labels are masked (-100)
    num_masked = (batch_output['labels'] == -100).sum().item()
    num_total = batch_output['labels'].numel()
    print(f"  - Masked tokens: {num_masked} / {num_total} ({100*num_masked/num_total:.1f}%)")

except Exception as e:
    print(f"✗ Data collator test failed: {e}")
    sys.exit(1)

# Test 6: Forward pass
print("\n[6/6] Testing forward pass...")
try:
    # Move inputs to model device
    inputs = {
        'input_ids': batch_output['input_ids'],
        'attention_mask': batch_output['attention_mask'],
        'pixel_values': batch_output['pixel_values'],
        'image_grid_thw': batch_output.get('image_grid_thw'),
        'labels': batch_output['labels'],
    }

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    assert hasattr(outputs, 'loss'), "Output missing loss"
    assert hasattr(outputs, 'logits'), "Output missing logits"

    print(f"✓ Forward pass successful")
    print(f"  - Loss: {outputs.loss.item():.4f}")
    print(f"  - Logits shape: {outputs.logits.shape}")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour Qwen3-VL integration is ready for training!")
print("\nNext steps:")
print("  1. Run training with 1 GPU for testing:")
print("     accelerate launch --config_file scripts/accelerate_configs/ddp.yaml \\")
print("         --num_processes 1 examples/qwen3_vl/sft.py --lora True \\")
print("         --per_device_train_batch_size 2")
print("\n  2. For full training with 8 GPUs:")
print("     accelerate launch --config_file scripts/accelerate_configs/fsdp.yaml \\")
print("         examples/qwen3_vl/sft.py --lora True")
print()
