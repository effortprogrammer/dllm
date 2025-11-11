# coding=utf-8
# Copyright 2025 dLLM team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3-VL model wrapper for diffusion language modeling"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers import (
    Qwen3VLForConditionalGeneration,
    PreTrainedModel,
)
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen3VLForMaskedLM(PreTrainedModel, GenerationMixin):
    base_model_prefix = "model"
    """
    Qwen3-VL model wrapper for masked diffusion language modeling.

    This class wraps Qwen3VLForConditionalGeneration to make it compatible with
    MDLMTrainer for diffusion-based training. The vision encoder and projection
    layers can be frozen during training.

    Key differences from the original Qwen3VL:
        - Returns MaskedLMOutput (with logits) instead of CausalLMOutput
        - Compatible with MDLMTrainer's masking mechanism
        - Supports freezing vision components (visual + merger)

    Example:
        >>> from transformers import AutoProcessor
        >>> from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM
        >>>
        >>> model = Qwen3VLForMaskedLM.from_pretrained(
        ...     "Qwen/Qwen3-VL-2B-Instruct",
        ...     freeze_vision=True,
        ...     freeze_merger=True
        ... )
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        >>>
        >>> # Forward pass (compatible with MDLMTrainer)
        >>> outputs = model(input_ids=input_ids, pixel_values=pixel_values, ...)
        >>> logits = outputs.logits  # (batch_size, seq_len, vocab_size)
    """

    def __init__(self, base_model: Qwen3VLForConditionalGeneration):
        # Initialize without config - we'll use the base model's config
        super().__init__(base_model.config)

        # Store the base Qwen3VL model
        self.model = base_model

        # Copy config attributes
        self.config = base_model.config

    def freeze_vision_components(self, freeze_visual: bool = True, freeze_merger: bool = True):
        """
        Freeze vision encoder and visual projection layers.

        Args:
            freeze_visual: Whether to freeze the vision encoder (visual model)
            freeze_merger: Whether to freeze the visual merger (projection layer)
        """
        frozen_params = 0
        total_params = 0

        # Freeze visual model (vision encoder)
        if freeze_visual and hasattr(self.model, 'visual'):
            for param in self.model.visual.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            logger.info("❄️  Froze vision encoder (visual model)")

        # Freeze visual merger (projection layer between vision and language)
        if freeze_merger and hasattr(self.model, 'merger'):
            for param in self.model.merger.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            logger.info("❄️  Froze visual merger (projection layer)")

        # Count total parameters
        for param in self.parameters():
            total_params += param.numel()

        if total_params == 0:
            logger.warning(
                "⚠️  No parameters detected while freezing vision components; "
                "this typically happens before DeepSpeed ZeRO-3 materializes weights. "
                "Skipping freeze statistics."
            )
            return

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            f"📊 Frozen {frozen_params:,} / {total_params:,} parameters "
            f"({100 * frozen_params / total_params:.2f}%)"
        )
        logger.info(
            f"🔥 Trainable: {trainable_params:,} / {total_params:,} parameters "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def get_input_embeddings(self):
        # For Qwen3VL, the embeddings are accessed via model.get_input_embeddings()
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # For Qwen3VL, set via model.set_input_embeddings()
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        # For Qwen3VL, access via model.get_output_embeddings()
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        # For Qwen3VL, set via model.set_output_embeddings()
        self.model.set_output_embeddings(new_embeddings)

    def get_base_model(self):
        """
        Ensure utilities like Liger see the underlying Qwen3VL base model.
        """
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        """
        Forward pass compatible with MDLMTrainer.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            pixel_values: Image pixel values from processor
            image_grid_thw: Image grid dimensions (temporal, height, width)
            labels: Target token IDs for loss computation (batch_size, seq_len)

        Returns:
            MaskedLMOutput with logits of shape (batch_size, seq_len, vocab_size)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through the base Qwen3VL model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Extract logits
        logits = outputs.logits

        # Compute loss if labels are provided
        # Note: MDLMTrainer will handle the actual loss computation with masking
        # This is just for compatibility
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load pretrained Qwen3VL model and wrap it for diffusion training.

        Args:
            pretrained_model_name_or_path: Model identifier from huggingface.co/models
                e.g., "Qwen/Qwen3-VL-2B-Instruct"
            freeze_vision: Whether to freeze vision encoder (default: False)
            freeze_merger: Whether to freeze projection layer (default: False)
            **kwargs: Additional arguments passed to Qwen3VLForConditionalGeneration

        Returns:
            Qwen3VLForMaskedLM instance
        """
        # Pop our custom arguments
        freeze_vision = kwargs.pop("freeze_vision", False)
        freeze_merger = kwargs.pop("freeze_merger", False)

        # Load the base Qwen3VL model
        logger.info(f"Loading Qwen3VL from {pretrained_model_name_or_path}...")

        # Add attn_implementation="eager" if not specified to avoid SDPA issues
        if "attn_implementation" not in kwargs:
            kwargs["attn_implementation"] = "eager"

        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Wrap it
        model = cls(base_model)

        # Optionally freeze vision components
        if freeze_vision or freeze_merger:
            model.freeze_vision_components(
                freeze_visual=freeze_vision,
                freeze_merger=freeze_merger
            )

        return model

    def save_pretrained(self, save_directory, **kwargs):
        """Save the wrapped model."""
        self.model.save_pretrained(save_directory, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate to the base model."""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()

    def resize_token_embeddings(self, new_num_tokens=None):
        """Resize token embeddings."""
        return self.model.resize_token_embeddings(new_num_tokens)
