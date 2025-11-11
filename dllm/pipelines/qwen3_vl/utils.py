# coding=utf-8
# Copyright 2025 dLLM team. All rights reserved.
"""Data collator and utilities for Qwen3-VL training"""

from dataclasses import dataclass
from typing import Any, Dict, List
import logging
import torch
from transformers import ProcessorMixin


logger = logging.getLogger(__name__)


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
        max_seq_length: Maximum sequence length (default: 4096)
        padding: Padding strategy ("max_length" or "longest")
    """

    processor: ProcessorMixin
    mask_prompt_loss: bool = True
    max_seq_length: int = 4096
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
        valid_records = []
        dropped_overlength = 0
        sample_records = []
        force_truncation = False

        for feature in features:
            image = feature.get("image", None)
            messages = feature.get("messages", [])

            qwen_messages = self._convert_to_qwen_format(messages, has_image=(image is not None))

            formatted_text = self.processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=False
            )

            length_check_kwargs = {
                "text": [formatted_text],
                "padding": False,
                "truncation": False,
                "return_tensors": "pt",
            }
            if image is not None:
                length_check_kwargs["images"] = [image]

            token_preview = self.processor(**length_check_kwargs)
            seq_len = token_preview["input_ids"].shape[-1]

            sample_records.append({
                "image": image,
                "messages": qwen_messages,
                "formatted_text": formatted_text,
                "seq_len": seq_len,
            })

        for record in sample_records:
            if self.max_seq_length is not None and record["seq_len"] > self.max_seq_length:
                dropped_overlength += 1
                continue
            valid_records.append(record)

        if not valid_records:
            # Fallback to the shortest sample to keep dataloader running, but warn loudly.
            shortest = min(sample_records, key=lambda r: r["seq_len"])
            force_truncation = True
            logger.warning(
                "All samples in the batch exceeded max_seq_length=%s. "
                "Keeping the shortest sample (len=%s) with truncation.",
                self.max_seq_length,
                shortest["seq_len"],
            )
            valid_records = [shortest]

        if dropped_overlength:
            logger.debug(
                "Dropped %d overlength samples (max_seq_length=%s).",
                dropped_overlength,
                self.max_seq_length,
            )

        images = [record["image"] for record in valid_records]
        texts = [record["messages"] for record in valid_records]

        formatted_texts = [record["formatted_text"] for record in valid_records]

        actual_images = [img for img in images if img is not None]

        if actual_images:
            # Process both text and images together
            # For image inputs, we need more tokens due to image placeholders
            # Either increase max_length or disable truncation to avoid mismatch
            processor_kwargs = {
                "text": formatted_texts,
                "images": actual_images,
                "padding": "longest" if not force_truncation else self.padding,
                "return_tensors": "pt",
            }

            # Keep truncation disabled so multimodal special tokens stay aligned with images.
            if force_truncation:
                processor_kwargs["max_length"] = self.max_seq_length
                processor_kwargs["truncation"] = True
            else:
                processor_kwargs["truncation"] = False

            batch_inputs = self.processor(**processor_kwargs)

            # If sequences are too long, manually truncate after processing
            if batch_inputs["input_ids"].shape[1] > self.max_seq_length:
                # Truncate all tensors to max_seq_length
                batch_inputs["input_ids"] = batch_inputs["input_ids"][:, :self.max_seq_length]
                batch_inputs["attention_mask"] = batch_inputs["attention_mask"][:, :self.max_seq_length]
                if "pixel_values" in batch_inputs:
                    # pixel_values shape is different, don't truncate
                    pass
                if "image_grid_thw" in batch_inputs:
                    # image_grid_thw is metadata, don't truncate
                    pass
        else:
            # No images, just process text
            text_kwargs = {
                "text": formatted_texts,
                "return_tensors": "pt",
            }
            if force_truncation:
                text_kwargs.update(
                    padding=self.padding,
                    max_length=self.max_seq_length,
                    truncation=True,
                )
            else:
                text_kwargs.update(
                    padding="longest",
                    truncation=False,
                )
            batch_inputs = self.processor(**text_kwargs)

        # If we processed without padding to max_length, pad manually now.
        if not force_truncation and self.padding == "max_length" and self.max_seq_length is not None:
            batch_inputs = self._pad_to_max_length(batch_inputs)

        # Create labels from input_ids
        labels = batch_inputs["input_ids"].clone()

        # Mask prompt loss if requested
        if self.mask_prompt_loss:
            labels = self._mask_prompt_tokens(texts, batch_inputs, labels)

        # Mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # IMPORTANT: Also mask vision/image tokens in labels
        # These tokens represent image features and should not be predicted
        vision_tokens = ["<|image_pad|>", "<|vision_start|>", "<|vision_end|>",
                        "<|vision_pad|>", "<|video_pad|>"]

        tokenizer = self.processor.tokenizer
        vocab = tokenizer.get_vocab()

        for token in vision_tokens:
            if token in vocab:
                token_id = tokenizer.convert_tokens_to_ids(token)
                labels[labels == token_id] = -100

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
        # Simplified approach: Use special tokens to find message boundaries
        # Look for the assistant marker tokens
        tokenizer = self.processor.tokenizer

        # For each example in the batch
        for idx in range(labels.shape[0]):
            # Get the token IDs for this example
            token_ids = batch_inputs["input_ids"][idx].tolist()

            # Find positions of assistant responses
            # Qwen3VL uses <|im_start|> and <|im_end|> tokens
            im_start_token = tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")

            # Also look for "assistant" as a substring in the tokenized text
            # Convert token IDs back to text to find role markers
            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)

            # Simple approach: Find "assistant" markers in the text
            # and map back to token positions
            assistant_marker = "<|im_start|>assistant"
            user_marker = "<|im_start|>user"

            # Find all occurrences of assistant responses
            import re
            assistant_spans = []
            for match in re.finditer(re.escape(assistant_marker), decoded_text):
                start = match.start()
                # Find the corresponding end marker
                end_match = decoded_text.find("<|im_end|>", start)
                if end_match != -1:
                    assistant_spans.append((start, end_match + len("<|im_end|>")))

            # Convert character positions to token positions (approximate)
            # This is a simplified approach
            if assistant_spans:
                # For simplicity, mask everything before the first assistant response
                first_assistant_pos = assistant_spans[0][0]
                # Estimate token position (rough approximation)
                chars_per_token = len(decoded_text) / len(token_ids)
                first_assistant_token_pos = int(first_assistant_pos / chars_per_token)

                # Mask all tokens before the assistant response
                if first_assistant_token_pos > 0:
                    labels[idx, :first_assistant_token_pos] = -100

        return labels

    def _pad_to_max_length(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Pad variable-length tensors in batch_inputs to max_seq_length.
        """
        if self.max_seq_length is None:
            return batch_inputs

        target = self.max_seq_length
        pad_id = self.processor.tokenizer.pad_token_id

        def _pad_tensor(tensor, pad_value=0):
            if tensor.shape[1] >= target:
                return tensor[:, :target]
            pad_width = target - tensor.shape[1]
            padding = (0, pad_width)
            return torch.nn.functional.pad(tensor, padding, value=pad_value)

        if "input_ids" in batch_inputs:
            batch_inputs["input_ids"] = _pad_tensor(batch_inputs["input_ids"], pad_id)
        if "attention_mask" in batch_inputs:
            batch_inputs["attention_mask"] = _pad_tensor(batch_inputs["attention_mask"], 0)

        return batch_inputs


def create_qwen3_vl_collator(processor, mask_prompt_loss=True, max_seq_length=4096):
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
