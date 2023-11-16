from typing import List

import pandas as pd
import torch


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