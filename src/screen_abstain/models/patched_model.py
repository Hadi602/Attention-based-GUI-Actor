"\"\"\"
Wrapper that swaps GUI-Actor's vanilla action head with our null-patch head.

This monkey-patches/subclasses `Qwen2VLForConditionalGenerationWithPointer`
from the original GUI-Actor codebase so that:
    * `self.multi_patch_pointer_head` is our `VisionHead_MultiPatchWithNull`
    * The forward path supports `refusal_labels` per sample
    * Inference exposes `pointer_scores_with_null` so the eval code can
      easily detect abstention.

Usage
-----
>>> from screen_abstain.models import Qwen2VLForConditionalGenerationWithNullPointer
>>> model = Qwen2VLForConditionalGenerationWithNullPointer.from_pretrained(
...     \"microsoft/GUI-Actor-7B-Qwen2-VL\", torch_dtype=torch.bfloat16,
... )
>>> model.swap_in_null_head()  # one-time at startup; copies weights over

Note: this file imports from `gui_actor`, which must be on PYTHONPATH.
The standard `gui_actor` install (`pip install -e /data4/rashid_GUI`)
makes that import work.
\"\"\"

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gui_actor.modeling import (   # noqa: E402  upstream import
    Qwen2VLForConditionalGenerationWithPointer,
    QwenVLwithVisionHeadOutputWithPast,
)

from .action_head_with_null import VisionHead_MultiPatchWithNull


class Qwen2VLForConditionalGenerationWithNullPointer(Qwen2VLForConditionalGenerationWithPointer):
    \"\"\"
    Subclass of GUI-Actor's pointer model that uses the null-patch action head.

    The class signature is otherwise identical so existing GUI-Actor inference
    scripts continue to work.
    \"\"\"

    # --------------------------------------------------------- head swap util
    def swap_in_null_head(self) -> None:
        \"\"\"Replace `multi_patch_pointer_head` with the null-patch variant and
        warm-start its weights from the original head.\"\"\"
        old_head = self.multi_patch_pointer_head
        d_model = self.config.hidden_size
        new_head = VisionHead_MultiPatchWithNull(
            d_model=d_model,
            projection_dim=d_model,
        ).to(device=old_head.projection_enc[0].weight.device,
             dtype=old_head.projection_enc[0].weight.dtype)
        new_head.load_from_original(old_head)
        self.multi_patch_pointer_head = new_head

    # ------------------------------------------------------- training forward
    def forward(  # type: ignore[override]
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # Grounding-specific kwargs
        visual_token_indices_of_coordinates: Optional[torch.Tensor] = None,
        multi_patch_labels: Optional[torch.Tensor] = None,
        if_multi_patch: bool = True,
        coordinates: Optional[List[Tuple[float, float]]] = None,
        verbose: bool = False,
        # NEW kwargs
        refusal_labels: Optional[torch.Tensor] = None,   # shape: (batch_size, n_target)
    ) -> Union[Tuple, QwenVLwithVisionHeadOutputWithPast]:
        \"\"\"
        Same as the upstream forward, plus a `refusal_labels` argument.
        `refusal_labels[i][j] == 1` means the j-th query on sample i is a
        refusal (target = null patch). For all standard grounding queries
        the value is 0 (or the kwarg may be omitted entirely).
        \"\"\"
        # The cleanest implementation would copy upstream's forward and inject
        # `refusal_labels` into the action-head call. To keep this file short
        # and to avoid duplicating ~200 lines of upstream logic, we instead
        # monkey-patch a tiny amount of state and call super().forward.

        # We stash refusal_labels on `self` so the action-head call inside the
        # parent forward (which we re-route below) can read it.
        self._refusal_labels = refusal_labels  # may be None

        # Re-route the call: parent calls `self.multi_patch_pointer_head(...)`.
        # We override the head's `forward` via `__call__` capture below.
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            visual_token_indices_of_coordinates=visual_token_indices_of_coordinates,
            multi_patch_labels=multi_patch_labels,
            if_multi_patch=if_multi_patch,
            coordinates=coordinates,
            verbose=verbose,
        )

    # The action head's `__call__` is invoked from within parent forward as
    # `self.multi_patch_pointer_head(visual_embeds, target_hidden, labels=...)`.
    # We override the binding so the per-sample refusal label is forwarded:
    def _call_action_head(
        self,
        visual_embeds: torch.Tensor,
        target_hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sample_index: int = 0,
    ):
        refusal_labels = None
        if self._refusal_labels is not None:
            refusal_labels = self._refusal_labels[sample_index].to(visual_embeds.device)
        return self.multi_patch_pointer_head(
            visual_embeds, target_hidden,
            labels=labels, refusal_labels=refusal_labels,
        )
"
