from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from preds_holder import PredsHolder


class Evaluator:
    def __init__(self, df: pd.DataFrame, threshold: float) -> None:
        self.__df = df
        self.__threshold = threshold

    @classmethod
    def constrant_using_preds_holder(cls, preds_holder: PredsHolder) -> "Evaluator":
        return cls(df=preds_holder.dataframe, threshold=preds_holder.threshold)

    def show_hist(self, save_path=None):
        bins, good_preds, bad_preds = self.__calcu_for_hist(df=self.__df)

        plt.figure()

        plt.hist(good_preds, bins=bins, alpha=0.5, label="True Negative", color="green", edgecolor="black", rwidth=0.8)
        plt.hist(bad_preds, bins=bins, alpha=0.5, label="True Positive", color="red", edgecolor="black", rwidth=0.8)
        plt.axvline(self.__threshold, color="k", linestyle="dashed", linewidth=1)

        plt.xlabel("Predictions")
        plt.ylabel("Frequency")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("Histogram of Predictions")

        if save_path:
            plt.savefig(f"{save_path}/hist.png", bbox_inches="tight")
        plt.show()

    def show_pdf(self, save_path=None):
        x_values, good_pdf, bad_pdf = self.__calcu_for_pdf(df=self.__df)

        plt.figure()

        plt.plot(x_values, bad_pdf, label="True Negative", color="green")
        plt.plot(x_values, good_pdf, label="True Positive", color="red")
        plt.axvline(self.__threshold, color="k", linestyle="dashed", linewidth=1)

        plt.ylim(bottom=0)
        plt.xlabel("Predictions")
        plt.ylabel("Probability Density Function")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("Probability Density Function of Predictions")

        if save_path:
            plt.savefig(f"{save_path}/pdf.png", bbox_inches="tight")
        plt.show()

    def show_hist_and_pdf(self, save_path=None):
        bins, good_preds, bad_preds = self.__calcu_for_hist(df=self.__df)
        x_values, good_pdf, bad_pdf = self.__calcu_for_pdf(df=self.__df)

        # Create a figure and a set of subplots
        _, ax1 = plt.subplots()

        # Hist
        ax1.hist(good_preds, bins=bins, alpha=0.5, label="True Negative Hist", color="green", edgecolor="black", rwidth=0.8)
        ax1.hist(bad_preds, bins=bins, alpha=0.5, label="True Positive Hist", color="red", edgecolor="black", rwidth=0.8)
        ax1.axvline(self.__threshold, color="k", linestyle="dashed", linewidth=1)

        ax1.set_xlabel("Predictions")
        ax1.set_ylabel("Frequency", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # PDF
        ax2 = ax1.twinx()
        ax2.plot(x_values, bad_pdf, label="True Negative PDF", color="green", linestyle="dashed")
        ax2.plot(x_values, good_pdf, label="True Positive PDF", color="red", linestyle="dashed")

        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("Probability Density Function", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        # Title and show
        ax1.legend(loc="center left", bbox_to_anchor=(1.2, 0.6))
        ax2.legend(loc="center left", bbox_to_anchor=(1.2, 0.4))
        plt.title("Histogram and Probability Density Function of Predictions")

        if save_path:
            plt.savefig(f"{save_path}/hist_and_pdf.png", bbox_inches="tight")
        plt.show()

    def show_confusion_matrix_and_metrics(self, save_path=None) -> pd.DataFrame:
        TP, TN, FP, FN = self.__culcu_for_confusion_matrix_and_metrics(df=self.__df)

        # Calculate FPR, FNR, TPR, TNR, and Accuracy
        FPR = FP / (TP + FP)
        FNR = FN / (FN + TP)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        Accuracy = (TP + TN) / (TP + FP + FN + TN)

        # Create the DataFrame
        df = pd.DataFrame(
            {
                "": ["Label Positive", "Label Negative", "False Rates"],
                "Pred Positive": [f"TP={TP}", f"FP={FP}", f"FPR={FPR:.2f}"],
                "Pred Negative": [f"FN={FN}", f"TN={TN}", f"FNR={FNR:.2f}"],
                "True Rates": [f"TPR={TPR:.2f}", f"TNR={TNR:.2f}", f"Accuracy={Accuracy:.2f}"],
            }
        ).set_index("")

        if save_path:
            df.to_csv(f"{save_path}/confusion_matrix_and_metrics_result.csv")
        return df

    @staticmethod
    def show_confusion_matrix_and_metrics_define(save_path=None) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "": ["Label Positive", "Label Negative", "False Rates"],
                "Pred Positive": ["True Positive (TP)", "False Positive (FP)", "FPR = FP / (TP + FP)"],
                "Pred Negative": ["False Negative (FN)", "True Negative (TN)", "FNR = FN / (FN + TP)"],
                "True Rates": ["TPR = TP / (TP + FN)", "TNR = TN / (TN + FP)", "Accuracy = (TP + TN) / (TP + FP + FN + TN)"],
            }
        ).set_index("")

        if save_path:
            df.to_csv(f"{save_path}/confusion_matrix_and_metrics_define.csv")
        return df

    @staticmethod
    def __calcu_for_hist(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_pred = df["pred"].min()
        max_pred = df["pred"].max()
        bins = np.linspace(min_pred, max_pred, 30)

        good_preds = df[df["label"] == 0]["pred"].values
        bad_preds = df[df["label"] == 1]["pred"].values
        return bins, good_preds, bad_preds

    def __calcu_for_pdf(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        good_df = df[df["label"] == 0]
        bad_df = df[df["label"] == 1]

        good_mean, good_std = self.__get_mean_and_std(values=good_df["pred"].values)
        bad_mean, bad_std = self.__get_mean_and_std(values=bad_df["pred"].values)
        x_values = np.linspace(min(df["pred"]), max(df["pred"]), 100)

        bad_pdf = stats.norm.pdf(x_values, bad_mean, bad_std)
        good_pdf = stats.norm.pdf(x_values, good_mean, good_std)

        return x_values, bad_pdf, good_pdf

    @staticmethod
    def __culcu_for_confusion_matrix_and_metrics(df: pd.DataFrame) -> Tuple[int, int, int, int]:
        labels = df["label"].values
        label_preds = df["label_pred"].values

        # Calculate TP, TN, FP, FN using NumPy
        TP = np.sum((labels == 1) & (label_preds == 1))
        TN = np.sum((labels == 0) & (label_preds == 0))
        FP = np.sum((labels == 0) & (label_preds == 1))
        FN = np.sum((labels == 1) & (label_preds == 0))

        return TP, TN, FP, FN

    @staticmethod
    def __get_mean_and_std(values: np.ndarray):
        descriptive_stats = stats.describe(values)
        return descriptive_stats.mean, np.sqrt(descriptive_stats.variance)
