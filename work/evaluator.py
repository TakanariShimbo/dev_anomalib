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
        bins, negative_preds, positive_preds = self.__calcu_for_hist(df=self.__df)

        plt.figure()

        plt.hist(negative_preds, bins=bins, alpha=0.5, label="Label Negative Hist", color="green", edgecolor="black", rwidth=0.8)
        plt.hist(positive_preds, bins=bins, alpha=0.5, label="Label Positive Hist", color="red", edgecolor="black", rwidth=0.8)
        plt.axvline(self.__threshold, color="k", linestyle="dashed", linewidth=1)

        plt.xlabel("Predictions")
        plt.ylabel("Frequency")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("Histogram of Predictions")

        if save_path:
            plt.savefig(f"{save_path}/hist.png", bbox_inches="tight")
        plt.show()

    def show_pdf(self, save_path=None):
        x_values, negative_pdf, positive_pdf = self.__calcu_for_pdf(df=self.__df)

        plt.figure()

        plt.axvline(self.__threshold, color="gray", linestyle="dashed", linewidth=1)

        plt.plot(x_values, negative_pdf, label="Label Negative PDF", color="green")
        plt.plot(x_values, positive_pdf, label="Label Positive PDF", color="red")

        plt.fill_between(x_values, negative_pdf, where=(x_values <= self.__threshold), color="lightgreen", alpha=0.4, label="True Negative Area")
        plt.fill_between(x_values, positive_pdf, where=(x_values >= self.__threshold), color="pink", alpha=0.4, label="True Positive Area")

        plt.fill_between(x_values, negative_pdf, where=(x_values >= self.__threshold), color="red", alpha=0.4, label="False Positive Area")
        plt.fill_between(x_values, positive_pdf, where=(x_values <= self.__threshold), color="green", alpha=0.4, label="False Negative Area")

        plt.ylim(bottom=0)
        plt.xlabel("Predictions")
        plt.ylabel("Probability Density Function")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("Probability Density Function of Predictions")

        if save_path:
            plt.savefig(f"{save_path}/pdf.png", bbox_inches="tight")
        plt.show()

    def show_hist_and_pdf(self, save_path=None):
        bins, negative_preds, positive_preds = self.__calcu_for_hist(df=self.__df)
        x_values, negative_pdf, positive_pdf = self.__calcu_for_pdf(df=self.__df)

        # Create a figure and a set of subplots
        _, ax1 = plt.subplots()

        # Hist
        ax1.hist(negative_preds, bins=bins, alpha=0.5, label="Label Negative Hist", color="green", edgecolor="black", rwidth=0.8)
        ax1.hist(positive_preds, bins=bins, alpha=0.5, label="Label Positive Hist", color="red", edgecolor="black", rwidth=0.8)
        ax1.axvline(self.__threshold, color="k", linestyle="dashed", linewidth=1)

        ax1.set_xlabel("Predictions")
        ax1.set_ylabel("Frequency", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # PDF
        ax2 = ax1.twinx()
        ax2.plot(x_values, negative_pdf, label="Label Negative PDF", color="green", linestyle="dashed")
        ax2.plot(x_values, positive_pdf, label="Label Positive PDF", color="red", linestyle="dashed")

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

        # Calculate metrics
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        FPR = 1.0 - TNR
        FNR = 1.0 - TPR
        precision = TP / (TP + FP)
        recall = TPR
        specificity = TNR
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        F1 = 2 * (precision * recall) / (precision + recall)

        df = pd.DataFrame(
            {
                "": [
                    f"Label Positive = {TP+FN}",
                    f"Label Negative = {TN+FP}",
                    "Metrics Other",
                ],
                f"Pred Positive = {TP+FP}": [
                    f"TP = {TP}",
                    f"FP = {FP}",
                    f"Precision = {precision:.2f}",
                ],
                f"Pred Negative = {TN+FN}": [
                    f"FN = {FN}",
                    f"TN = {TN}",
                    "",
                ],
                "Metrics True Rate": [
                    f"TPR = {TPR:.2f}",
                    f"TNR = {TNR:.2f}",
                    f"Accuracy = {accuracy:.2f}",
                ],
                "Metrics False Rate": [
                    f"FNR = {FNR:.2f}",
                    f"FPR = {FPR:.2f}",
                    f"MCC = {(TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))**(1/2):.2f}",
                ],
                "Metrics Other": [
                    f"Recall = {recall:.2f}",
                    f"Specificity = {specificity:.2f}",
                    f"F1 = {F1:.2f}",
                ],
            }
        ).set_index("")

        if save_path:
            df.to_csv(f"{save_path}/confusion_matrix_and_metrics_result.csv")
        return df


    @staticmethod
    def show_confusion_matrix_and_metrics_define(save_path=None) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "": [
                    "Label Positive",
                    "Label Negative",
                    "Metrics Other",
                ],
                "Pred Positive": [
                    "True Positive (TP)",
                    "False Positive (FP)",
                    "Precision = TP / (TP + FP)",
                ],
                "Pred Negative": [
                    "False Negative (FN)",
                    "True Negative (TN)",
                    "",
                ],
                "Metrics True Rate": [
                    "TPR = TP / (TP + FN)",
                    "TNR = TN / (TN + FP)",
                    "Accuracy = (TP + TN) / (TP + FP + FN + TN)",
                ],
                "Metrics False Rate": [
                    "FNR = 1 - TPR",
                    "FPR = 1 - TPR",
                    "MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))",
                ],
                "Metrics Other": [
                    "Recall = TPR",
                    "Specificity = TNR",
                    "F1 = 2 * (Precision * Recall) / (Precision + Recall)",
                ],
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

        negative_preds = df[df["label"] == 0]["pred"].values
        positive_preds = df[df["label"] == 1]["pred"].values
        return bins, negative_preds, positive_preds

    def __calcu_for_pdf(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        negative_df = df[df["label"] == 0]
        positive_df = df[df["label"] == 1]

        total_samples = len(df)
        weight_negative = len(negative_df) / total_samples
        weight_positive = len(positive_df) / total_samples

        negative_mean, negative_std = self.__get_mean_and_std(values=negative_df["pred"].values)
        positive_mean, positive_std = self.__get_mean_and_std(values=positive_df["pred"].values)
        x_values = np.linspace(min(df["pred"]), max(df["pred"]), 100)

        negative_pdf = stats.norm.pdf(x_values, negative_mean, negative_std) * weight_negative
        positive_pdf = stats.norm.pdf(x_values, positive_mean, positive_std) * weight_positive

        return x_values, negative_pdf, positive_pdf

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
