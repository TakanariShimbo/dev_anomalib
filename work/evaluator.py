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
        
        plt.hist(good_preds, bins=bins, alpha=0.5, label='Good', color='green', edgecolor='black', rwidth=0.8)
        plt.hist(bad_preds, bins=bins, alpha=0.5, label='Bad', color='red', edgecolor='black', rwidth=0.8)
        plt.axvline(self.__threshold, color='k', linestyle='dashed', linewidth=1)

        plt.xlabel('Predictions')
        plt.ylabel('Frequency')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('Histogram of Predictions')

        if save_path:
            plt.savefig(f"{save_path}/hist.png", bbox_inches='tight')
        plt.show()

    def show_pdf(self, save_path=None):
        x_values, good_pdf, bad_pdf = self.__calcu_for_pdf(df=self.__df)

        plt.figure()

        plt.plot(x_values, bad_pdf, label='Good', color='green')
        plt.plot(x_values, good_pdf, label='Bad', color='red')
        plt.axvline(self.__threshold, color='k', linestyle='dashed', linewidth=1)
        
        plt.ylim(bottom=0)
        plt.xlabel('Predictions')
        plt.ylabel('Probability Density Function')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('Probability Density Function of Predictions')

        if save_path:
            plt.savefig(f"{save_path}/pdf.png", bbox_inches='tight')
        plt.show()

    def show_hist_and_pdf(self, save_path=None):
        bins, good_preds, bad_preds = self.__calcu_for_hist(df=self.__df)
        x_values, good_pdf, bad_pdf = self.__calcu_for_pdf(df=self.__df)

        # Create a figure and a set of subplots
        _, ax1 = plt.subplots()

        # Hist
        ax1.hist(good_preds, bins=bins, alpha=0.5, label='Good Hist', color='green', edgecolor='black', rwidth=0.8)
        ax1.hist(bad_preds, bins=bins, alpha=0.5, label='Bad Hist', color='red', edgecolor='black', rwidth=0.8)
        ax1.axvline(self.__threshold, color='k', linestyle='dashed', linewidth=1)

        ax1.set_xlabel('Predictions')
        ax1.set_ylabel('Frequency', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # PDF
        ax2 = ax1.twinx()
        ax2.plot(x_values, bad_pdf, label='Good PDF', color='green', linestyle='dashed')
        ax2.plot(x_values, good_pdf, label='Bad PDF', color='red', linestyle='dashed')
        
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel('Probability Density Function', color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

        # Title and show
        ax1.legend(loc='center left', bbox_to_anchor=(1.2, 0.6))
        ax2.legend(loc='center left', bbox_to_anchor=(1.2, 0.4))
        plt.title('Histogram and Probability Density Function of Predictions')

        if save_path:
            plt.savefig(f"{save_path}/hist_and_pdf.png", bbox_inches='tight')
        plt.show()

    @staticmethod
    def __calcu_for_hist(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_pred = df['pred'].min()
        max_pred = df['pred'].max()
        bins = np.linspace(min_pred, max_pred, 30)

        good_preds = df[df['label'] == 0]['pred'].values
        bad_preds = df[df['label'] == 1]['pred'].values
        return bins, good_preds, bad_preds

    def __calcu_for_pdf(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        good_df = df[df['label'] == 0]
        bad_df = df[df['label'] == 1]

        good_mean, good_std = self.__get_mena_and_std(values=good_df['pred'].values)
        bad_mean, bad_std = self.__get_mena_and_std(values=bad_df['pred'].values)
        x_values = np.linspace(min(df['pred']), max(df['pred']), 100)

        bad_pdf = stats.norm.pdf(x_values, bad_mean, bad_std)
        good_pdf = stats.norm.pdf(x_values, good_mean, good_std)

        return x_values, bad_pdf, good_pdf
    
    @staticmethod
    def __get_mena_and_std(values: np.ndarray):
        descriptive_stats = stats.describe(values)
        return descriptive_stats.mean, np.sqrt(descriptive_stats.variance)
    


if __name__ == "__main__":
    np.random.seed(0)

    # Generate data
    n_samples = 50
    data_1 = np.random.normal(loc=0.25, scale=0.05, size=n_samples)
    data_2 = np.random.normal(loc=0.75, scale=0.05, size=n_samples)

    # Labels (0 for the first distribution, 1 for the second)
    labels_1 = np.zeros(n_samples, dtype=int)
    labels_2 = np.ones(n_samples, dtype=int)

    # Combine the data
    preds = np.concatenate([data_1, data_2])
    labels = np.concatenate([labels_1, labels_2])

    # Create & Shuffle dataFrame
    df = pd.DataFrame({'pred': preds, 'label': labels})
    df = df.sample(frac=1).reset_index(drop=True)

    # Threshold for predictions
    threshold = 0.5

    evaluator = Evaluator(df, threshold)

    # Now you can use the evaluator to show the plots
    evaluator.show_hist()
    evaluator.show_pdf()
    evaluator.show_hist_and_pdf()