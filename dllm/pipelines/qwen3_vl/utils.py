# coding=utf-8
# Copyright 2025 dLLM team. All rights reserved.
"""Data collator and utilities for Qwen3-VL training"""

from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from transformers import ProcessorMixin


@dataclass
class Qwen3VLDataCollator:
    """
    Data collator for Qwen3-VL that handles multi-modal inputs (images + text).

    This collator:
    1. Uses AutoProcessor to handle image and text processing
    2. Applies chat template for conversation formatting
    3. Handles prompt loss masking for instruction tuning
    4. Creates batched inputs compatible with Qwen3VLForMaskedLM

    Args:
        processor: Qwen3VL AutoProcessor for handling images and text
        mask_prompt_loss: Whether to mask prompt (user) tokens in loss computation
        max_seq_length: Maximum sequence length (default: 2048)
        padding: Padding strategy ("max_length" or "longest")
    """

    processor: ProcessorMixin
    mask_prompt_loss: bool = True
    max_seq_length: int = 2048
    padding: str = "max_length"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples with images and text.

        Args:
            features: List of dicts with keys:
                - "messages": list of message dicts (OpenAI format)
                - "image": PIL Image

        Returns:
            Dict with keys:
                - input_ids: (batch, seq_len)
                - attention_mask: (batch, seq_len)
                - labels: (batch, seq_len) with -100 for masked positions
                - pixel_values: (batch, channels, height, width) or None
                - image_grid_thw: (batch, 3) with (temporal, height, width) or None
        """
        # Separate images and messages
        images = []
        texts = []

        for feature in features:
            image = feature.get("image", None)
            messages = feature.get("messages", [])

            # Convert messages to Qwen3VL format
            # Qwen3VL expects: [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}, ...]
            qwen_messages = self._convert_to_qwen_format(messages, has_image=(image is not None))

            images.append(image)
            texts.append(qwen_messages)

        # Process with Qwen3VL processor
        # This handles both text tokenization and image preprocessing
        batch_inputs = self.processor.apply_chat_template(
            texts,
            images=images if any(img is not None for img in images) else None,
            tokenize=True,
            add_generation_prompt=False,  # We don't want generation prompt for training
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Create labels from input_ids
        labels = batch_inputs["input_ids"].clone()

        # Mask prompt loss if requested
        if self.mask_prompt_loss:
            labels = self._mask_prompt_tokens(texts, batch_inputs, labels)

        # Mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Add labels to batch
        batch_inputs["labels"] = labels

        return batch_inputs

    def _convert_to_qwen_format(
        self, messages: List[Dict[str, str]], has_image: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style messages to Qwen3VL format.

        Input (from our dataset):
            [{"role": "user", "content": "text"}, {"role": "assistant", "content": "text"}]

        Output (Qwen3VL format):
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "text"}]},
             {"role": "assistant", "content": [{"type": "text", "text": "text"}]}]
        """
        qwen_messages = []

        for i, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            # Build content list
            content_list = []

            # Add image to first user message
            if role == "user" and i == 0 and has_image:
                content_list.append({"type": "image"})

            # Add text
            content_list.append({"type": "text", "text": content})

            qwen_messages.append({
                "role": role,
                "content": content_list
            })

        return qwen_messages

    def _mask_prompt_tokens(
        self,
        texts: List[List[Dict[str, Any]]],
        batch_inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask prompt (user) tokens in labels to only compute loss on assistant responses.

        This is done by:
        1. Processing each conversation separately to find assistant response positions
        2. Masking (setting to -100) all tokens before and including the last user message

        Args:
            texts: List of conversations in Qwen format
            batch_inputs: Batch inputs from processor
            labels: Initial labels (copy of input_ids)

        Returns:
            Labels with prompt tokens masked
        """
        # For each example in the batch
        for idx, messages in enumerate(texts):
            # Find positions of assistant messages
            # We need to mask everything except assistant responses

            # Simple heuristic: tokenize each message separately and find positions
            # This is approximate but works for most cases
            current_pos = 0

            for msg_idx, msg in enumerate(messages):
                role = msg["role"]

                # Tokenize this message to estimate its length
                # Note: Don't pass images=None as it causes duplicate keyword argument error
                msg_tokens = self.processor.apply_chat_template(
                    [msg],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors="pt",
                )["input_ids"]

                msg_length = msg_tokens.shape[1]

                # If this is a user message, mask it
                if role == "user":
                    end_pos = min(current_pos + msg_length, labels.shape[1])
                    labels[idx, current_pos:end_pos] = -100

                current_pos += msg_length

                # Stop if we exceed sequence length
                if current_pos >= labels.shape[1]:
                    break

        return labels


def create_qwen3_vl_collator(processor, mask_prompt_loss=True, max_seq_length=2048):
    """
    Factory function to create Qwen3VL data collator.

    Args:
        processor: Qwen3VL AutoProcessor
        mask_prompt_loss: Whether to mask prompt tokens in loss
        max_seq_length: Maximum sequence length

    Returns:
        Qwen3VLDataCollator instance
    """
    return Qwen3VLDataCollator(
        processor=processor,
        mask_prompt_loss=mask_prompt_loss,
        max_seq_length=max_seq_length,
    )
