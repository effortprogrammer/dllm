# coding=utf-8
# Copyright 2025 dLLM team. All rights reserved.
"""Trainer for Qwen3-VL with diffusion language modeling"""

import torch
from typing import Any, Dict, Optional
from dllm.core.trainers import MDLMTrainer


class Qwen3VLTrainer(MDLMTrainer):
    """
    Trainer for Qwen3-VL with masked diffusion language modeling.

    This extends MDLMTrainer to handle multi-modal inputs (images + text).
    The main differences are:
        - Handles pixel_values and image_grid_thw tensors
        - Ensures proper device placement for vision inputs
        - Compatible with Qwen3VLDataCollator outputs

    Example:
        >>> from transformers import AutoProcessor
        >>> from dllm.pipelines.qwen3_vl import Qwen3VLForMaskedLM, Qwen3VLTrainer
        >>> from dllm.pipelines.qwen3_vl.utils import Qwen3VLDataCollator
        >>>
        >>> model = Qwen3VLForMaskedLM.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        >>> collator = Qwen3VLDataCollator(processor=processor)
        >>>
        >>> trainer = Qwen3VLTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     data_collator=collator,
        ... )
        >>> trainer.train()
    """

    def _preprocess_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Preprocess inputs to ensure proper device placement for multi-modal data.

        This handles:
            - pixel_values: Image tensors
            - image_grid_thw: Image grid dimensions (temporal, height, width)
            - video_grid_thw: Video grid dimensions (if present)
            - rope_deltas: RoPE position deltas (if present)

        Args:
            inputs: Dictionary of input tensors (modified in-place)
        """
        # Move vision-related tensors to the correct device
        vision_keys = ["pixel_values", "image_grid_thw", "video_grid_thw", "rope_deltas"]

        for key in vision_keys:
            if key in inputs and inputs[key] is not None:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.args.device)

    def _postprocess_outputs(self, outputs: Any) -> None:
        """
        Postprocess outputs if needed.

        Currently no postprocessing is required for Qwen3VL,
        but this hook is available for future extensions.

        Args:
            outputs: Model outputs
        """
        pass

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute the masked diffusion loss for multi-modal inputs.

        This method:
        1. Preprocesses inputs (device placement for vision tensors)
        2. Calls parent MDLMTrainer.compute_loss() which:
           - Samples diffusion timesteps
           - Applies stochastic masking (excluding positions with label=-100)
           - Computes weighted cross-entropy loss
        3. Returns loss (and optionally outputs)

        Note: Image tokens are already marked with -100 in labels by the data collator,
        so they will not be masked during diffusion training.

        Args:
            model: Qwen3VLForMaskedLM model
            inputs: Dictionary with keys:
                - input_ids: (batch, seq_len)
                - attention_mask: (batch, seq_len)
                - labels: (batch, seq_len) with -100 for masked positions
                - pixel_values: (batch, channels, height, width) or None
                - image_grid_thw: (batch, 3) or None
            return_outputs: Whether to return model outputs
            **kwargs: Additional arguments

        Returns:
            Loss tensor (and optionally model outputs)
        """
        # Preprocess inputs (device placement)
        self._preprocess_inputs(inputs)

        # Call parent's compute_loss which handles the diffusion logic
        # The parent will:
        # 1. Sample timesteps t
        # 2. Apply masking according to α(t) (but not to positions with label=-100)
        # 3. Forward through model
        # 4. Compute weighted loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def prediction_step(
        self,
        model,
        inputs: Dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ):
        """
        Perform a prediction step for evaluation.

        This ensures vision tensors are on the correct device before evaluation.

        Args:
            model: The model
            inputs: Input dictionary
            prediction_loss_only: Whether to return only loss
            ignore_keys: Keys to ignore in outputs

        Returns:
            Tuple of (loss, logits, labels)
        """
        # Preprocess inputs
        self._preprocess_inputs(inputs)

        # Call parent's prediction_step
        return super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
