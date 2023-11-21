from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from preds_holder import PredsHolder


class Evaluator:
    def __init__(self, df: pd.DataFrame, threshold: float) -> None:
        self.__df = df
        self.threshold = threshold

        self.__bins, self.__negative_preds, self.__positive_preds = self.__calcu_for_hist(df=self.__df)
        self.__x_values, self.__negative_pdf, self.__positive_pdf = self.__calcu_for_pdf(df=self.__df)

        (
            self.true_positive,
            self.true_negative,
            self.false_positive,
            self.false_negative,
            self.true_positive_rate,
            self.true_negative_rate,
            self.false_positive_rate,
            self.false_negative_rate,
            self.precision,
            self.recall,
            self.specificity,
            self.accuracy,
            self.f1_score,
            self.matthews_correlation_coefficient,
        ) = self.__culcu_for_confusion_matrix_and_metrics(df=self.__df)

    @classmethod
    def constrant_using_preds_holder(cls, preds_holder: PredsHolder) -> "Evaluator":
        return cls(df=preds_holder.dataframe, threshold=preds_holder.threshold)

    def show_hist(self, save_path=None):
        plt.figure()

        plt.hist(self.__negative_preds, bins=self.__bins, alpha=0.5, label="Label Negative Hist", color="green", edgecolor="black", rwidth=0.8)
        plt.hist(self.__positive_preds, bins=self.__bins, alpha=0.5, label="Label Positive Hist", color="red", edgecolor="black", rwidth=0.8)
        plt.axvline(self.threshold, color="gray", linestyle="dashed", linewidth=1)

        plt.xlabel("Predictions")
        plt.ylabel("Frequency")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title(f"Histogram of Predictions TS:{self.threshold:.2f}")

        if save_path:
            plt.savefig(f"{save_path}/hist_{self.threshold:.2f}.png", bbox_inches="tight")
        plt.show()

    def show_pdf(self, save_path=None):
        plt.figure()

        plt.axvline(self.threshold, color="gray", linestyle="dashed", linewidth=1)

        plt.plot(self.__x_values, self.__negative_pdf, label="Label Negative PDF", color="green")
        plt.plot(self.__x_values, self.__positive_pdf, label="Label Positive PDF", color="red")

        plt.fill_between(self.__x_values, self.__negative_pdf, where=(self.__x_values <= self.threshold), color="lightgreen", alpha=0.4, label="True Negative Area")
        plt.fill_between(self.__x_values, self.__positive_pdf, where=(self.__x_values >= self.threshold), color="pink", alpha=0.4, label="True Positive Area")

        plt.fill_between(self.__x_values, self.__negative_pdf, where=(self.__x_values >= self.threshold), color="red", alpha=0.4, label="False Positive Area")
        plt.fill_between(self.__x_values, self.__positive_pdf, where=(self.__x_values <= self.threshold), color="green", alpha=0.4, label="False Negative Area")

        plt.ylim(bottom=0)
        plt.xlabel("Predictions")
        plt.ylabel("Probability Density Function")

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title(f"Probability Density Function of Predictions TS:{self.threshold:.2f}")

        if save_path:
            plt.savefig(f"{save_path}/pdf_{self.threshold:.2f}.png", bbox_inches="tight")
        plt.show()

    def show_hist_and_pdf(self, save_path=None):
        # Create a figure and a set of subplots
        _, ax1 = plt.subplots()

        # Hist
        ax1.hist(self.__negative_preds, bins=self.__bins, alpha=0.5, label="Label Negative Hist", color="green", edgecolor="black", rwidth=0.8)
        ax1.hist(self.__positive_preds, bins=self.__bins, alpha=0.5, label="Label Positive Hist", color="red", edgecolor="black", rwidth=0.8)
        ax1.axvline(self.threshold, color="gray", linestyle="dashed", linewidth=1)

        ax1.set_xlabel("Predictions")
        ax1.set_ylabel("Frequency", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # PDF
        ax2 = ax1.twinx()
        ax2.plot(self.__x_values, self.__negative_pdf, label="Label Negative PDF", color="green", linestyle="dashed")
        ax2.plot(self.__x_values, self.__positive_pdf, label="Label Positive PDF", color="red", linestyle="dashed")

        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("Probability Density Function", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        # Title and show
        ax1.legend(loc="center left", bbox_to_anchor=(1.2, 0.6))
        ax2.legend(loc="center left", bbox_to_anchor=(1.2, 0.4))
        plt.title(f"Histogram and Probability Density Function of Predictions TS:{self.threshold:.2f}")

        if save_path:
            plt.savefig(f"{save_path}/hist_and_pdf_{self.threshold:.2f}.png", bbox_inches="tight")
        plt.show()

    def show_confusion_matrix_and_metrics(self, save_path=None) -> None:
        df = pd.DataFrame(
            {
                "": [
                    f"Label Positive = {self.true_positive+self.false_negative}",
                    f"Label Negative = {self.true_negative+self.false_positive}",
                    "Metrics Other",
                ],
                f"Pred Positive = {self.true_positive+self.false_positive}": [
                    f"TP = {self.true_positive}",
                    f"FP = {self.false_positive}",
                    f"Precision = {self.precision:.2f}",
                ],
                f"Pred Negative = {self.true_negative+self.false_negative}": [
                    f"FN = {self.false_negative}",
                    f"TN = {self.true_negative}",
                    "",
                ],
                "Metrics True Rate": [
                    f"TPR = {self.true_positive_rate:.2f}",
                    f"TNR = {self.true_negative_rate:.2f}",
                    f"Accuracy = {self.accuracy:.2f}",
                ],
                "Metrics False Rate": [
                    f"FNR = {self.false_negative_rate:.2f}",
                    f"FPR = {self.false_positive_rate:.2f}",
                    f"MCC = {self.matthews_correlation_coefficient:.2f}",
                ],
                "Metrics Other": [
                    f"Recall = {self.recall:.2f}",
                    f"Specificity = {self.specificity:.2f}",
                    f"F1 = {self.f1_score:.2f}",
                ],
            }
        ).set_index("")

        _, ax = plt.subplots(figsize=(10,2))
        ax.axis("off")
        col_text = df.reset_index().values
        col_labels = [''] + list(df.columns)
        ax.table(cellText=col_text, colLabels=col_labels, loc="center", bbox=[0, 0, 1, 1])

        plt.title(f"Confusion Matrix and Metrics TS:{self.threshold:.2f}")

        if save_path:
            plt.savefig(f"{save_path}/confusion_matrix_and_metrics_{self.threshold:.2f}.png", bbox_inches="tight")
        plt.show()

    @staticmethod
    def show_confusion_matrix_and_metrics_define(save_path=None) -> None:
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

        _, ax = plt.subplots(figsize=(40,2))
        ax.axis("off")
        col_text = df.reset_index().to_numpy()
        col_labels = [''] + list(df.columns)
        ax.table(cellText=col_text, colLabels=col_labels, loc="center", bbox=[0, 0, 1, 1])

        plt.title("Confusion Matrix and Metrics Define")

        if save_path:
            plt.savefig(f"{save_path}/confusion_matrix_and_metrics_define.png", bbox_inches="tight")
        plt.show()

    @staticmethod
    def __calcu_for_hist(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_pred = df["pred"].min()
        max_pred = df["pred"].max()
        bins = np.linspace(min_pred, max_pred, 30)

        negative_preds = df[df["label"] == 0]["pred"].to_numpy()
        positive_preds = df[df["label"] == 1]["pred"].to_numpy()
        return bins, negative_preds, positive_preds

    def __calcu_for_pdf(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        negative_df = df[df["label"] == 0]
        positive_df = df[df["label"] == 1]

        total_samples = len(df)
        weight_negative = len(negative_df) / total_samples
        weight_positive = len(positive_df) / total_samples

        negative_mean, negative_std = self.__get_mean_and_std(values=negative_df["pred"].to_numpy())
        positive_mean, positive_std = self.__get_mean_and_std(values=positive_df["pred"].to_numpy())
        x_values = np.linspace(min(df["pred"]), max(df["pred"]), 100)

        negative_pdf = stats.norm.pdf(x_values, negative_mean, negative_std) * weight_negative
        positive_pdf = stats.norm.pdf(x_values, positive_mean, positive_std) * weight_positive

        return x_values, negative_pdf, positive_pdf

    @staticmethod
    def __culcu_for_confusion_matrix_and_metrics(
        df: pd.DataFrame,
    ) -> Tuple[int, int, int, int, float, float, float, float, float, float, float, float, float, float]:
        labels = df["label"].to_numpy()
        label_preds = df["pred_label"].to_numpy()

        true_positive = np.sum((labels == 1) & (label_preds == 1))
        true_negative = np.sum((labels == 0) & (label_preds == 0))
        false_positive = np.sum((labels == 0) & (label_preds == 1))
        false_negative = np.sum((labels == 1) & (label_preds == 0))

        true_positive_rate = true_positive / (true_positive + false_negative)
        true_negative_rate = true_negative / (true_negative + false_positive)
        false_positive_rate = 1.0 - true_negative_rate
        false_negative_rate = 1.0 - true_positive_rate
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive_rate
        specificity = true_negative_rate
        accuracy = (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)
        matthews_correlation_coefficient = (true_positive * true_negative - false_positive * false_negative) / (
            (true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative)
        ) ** (1 / 2)

        return (
            true_positive,
            true_negative,
            false_positive,
            false_negative,
            true_positive_rate,
            true_negative_rate,
            false_positive_rate,
            false_negative_rate,
            precision,
            recall,
            specificity,
            accuracy,
            f1_score,
            matthews_correlation_coefficient,
        )

    @staticmethod
    def __get_mean_and_std(values: np.ndarray):
        descriptive_stats = stats.describe(values)
        return descriptive_stats.mean, np.sqrt(descriptive_stats.variance)
