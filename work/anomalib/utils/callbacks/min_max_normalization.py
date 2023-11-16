"""Anomaly Score Normalization Callback that uses min-max normalization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, List

import pandas as pd

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule
from anomalib.post_processing.normalization.min_max import normalize
from anomalib.utils.metrics import MinMax


class PredsHolder:
    def __init__(self) -> None:
        self.__IMAGE_PATH_COLUMN = "image_path"
        self.__PRED_COLUMN = "pred"
        self.__PRED_NORM_COLUMN = "pred_norm"
        self.__PRED_LABEL_COLUMN = "pred_label"
        self.__LABEL_COLUMN = "label"
        self.__threshold = 0.0
        self.__min = 0.0
        self.__max = 0.0
        # Adding the new columns to the DataFrame
        self.__df = pd.DataFrame(columns=[
            self.__IMAGE_PATH_COLUMN, 
            self.__PRED_COLUMN, 
            self.__PRED_NORM_COLUMN, 
            self.__PRED_LABEL_COLUMN, 
            self.__LABEL_COLUMN
        ])

    def reset(self):
        self.__df = pd.DataFrame(columns=[
            self.__IMAGE_PATH_COLUMN, 
            self.__PRED_COLUMN, 
            self.__PRED_NORM_COLUMN, 
            self.__PRED_LABEL_COLUMN, 
            self.__LABEL_COLUMN
        ])

    def add_multiple(self, image_paths: List[str], preds: torch.tensor, pred_norms: torch.tensor, pred_labels: torch.tensor, labels: torch.tensor) -> None:
        preds = preds.tolist()
        pred_norms = pred_norms.tolist()
        pred_labels = pred_labels.tolist()
        labels = labels.tolist()

        if not (len(image_paths) == len(preds) == len(pred_norms) == len(pred_labels) == len(labels)):
            raise ValueError("All lists must be of the same length.")

        new_rows = pd.DataFrame({
            self.__IMAGE_PATH_COLUMN: image_paths,
            self.__PRED_COLUMN: preds,
            self.__PRED_NORM_COLUMN: pred_norms,
            self.__PRED_LABEL_COLUMN: pred_labels,
            self.__LABEL_COLUMN: labels
        })

        self.__df = pd.concat([self.__df, new_rows], ignore_index=True)
    
    def set_thresholds(self, threshold: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
        self.__threshold = threshold.item()
        self.__min = min_val.item()
        self.__max = max_val.item()
        
    @property
    def dataframe(self) -> pd.DataFrame:
        self.__df[self.__PRED_LABEL_COLUMN] = self.__df[self.__PRED_LABEL_COLUMN].astype(int)
        return self.__df
    
    @property
    def threshold(self) -> float:
        return self.__threshold

    @property
    def min(self) -> float:
        return self.__min

    @property
    def max(self) -> float:
        return self.__max



class MinMaxNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__preds_holder = PredsHolder()
    
    def setup(self, trainer: pl.Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
        """Adds min_max metrics to normalization metrics."""
        del trainer, stage  # These variables are not used.

        if not hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics = MinMax().cpu()
        elif not isinstance(pl_module.normalization_metrics, MinMax):
            raise AttributeError(
                f"Expected normalization_metrics to be of type MinMax, got {type(pl_module.normalization_metrics)}"
            )

    def on_test_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        del trainer  # `trainer` variable is not used.

        for metric in (pl_module.image_metrics, pl_module.pixel_metrics):
            if metric is not None:
                metric.set_threshold(0.5)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, update the min and max observed values."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        if "anomaly_maps" in outputs:
            pl_module.normalization_metrics(outputs["anomaly_maps"])
        elif "box_scores" in outputs:
            pl_module.normalization_metrics(torch.cat(outputs["box_scores"]))
        elif "pred_scores" in outputs:
            pl_module.normalization_metrics(outputs["pred_scores"])
        else:
            raise ValueError("No values found for normalization, provide anomaly maps, bbox scores, or image scores")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    def _normalize_batch(self, outputs, pl_module) -> None:
        """Normalize a batch of predictions."""
        image_threshold = pl_module.image_threshold.value.cpu()
        pixel_threshold = pl_module.pixel_threshold.value.cpu()
        stats = pl_module.normalization_metrics.cpu()

        preds = outputs["pred_scores"]
        norm_preds = normalize(preds, image_threshold, stats.min, stats.max)
        outputs["pred_scores"] = norm_preds

        self.__preds_holder.add_multiple(
            image_paths=outputs["image_path"],
            preds=preds,
            pred_norms=norm_preds,
            pred_labels=outputs["pred_labels"],
            labels=outputs["label"],
        )
        self.__preds_holder.set_thresholds(
            threshold=image_threshold,
            min_val=stats.min,
            max_val=stats.max
        )

        if "anomaly_maps" in outputs:
            outputs["anomaly_maps"] = normalize(outputs["anomaly_maps"], pixel_threshold, stats.min, stats.max)
        if "box_scores" in outputs:
            outputs["box_scores"] = [
                normalize(scores, pixel_threshold, stats.min, stats.max) for scores in outputs["box_scores"]
            ]

    def get_preds_holder(self) -> PredsHolder:
        return self.__preds_holder