from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats 
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, df: pd.DataFrame) -> None:
        self.__df = df
    
    def show_hist(self):
        bins, good_preds, bad_preds = self.__calcu_for_hist(df=self.__df)
        
        plt.figure()
        
        plt.hist(good_preds, bins=bins, alpha=0.5, label='Good', color='green', edgecolor='black', rwidth=0.8)
        plt.hist(bad_preds, bins=bins, alpha=0.5, label='Bad', color='red', edgecolor='black', rwidth=0.8)

        plt.xlabel('Predictions')
        plt.ylabel('Frequency')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('Histogram of Predictions')
        plt.show()

    def show_pdf(self):
        x_values, good_pdf, bad_pdf = self.__calcu_for_pdf(df=self.__df)

        plt.figure()

        plt.plot(x_values, bad_pdf, label='Good', color='green')
        plt.plot(x_values, good_pdf, label='Bad', color='red')
        
        plt.ylim(bottom=0)
        plt.xlabel('Predictions')
        plt.ylabel('Probability Density Function')

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('Probability Density Function of Predictions')
        plt.show()

    def show_hist_and_pdf(self):
        bins, good_preds, bad_preds = self.__calcu_for_hist(df=self.__df)
        x_values, good_pdf, bad_pdf = self.__calcu_for_pdf(df=self.__df)

        # Create a figure and a set of subplots
        _, ax1 = plt.subplots()

        # Hist
        ax1.hist(good_preds, bins=bins, alpha=0.5, label='Good Hist', color='green', edgecolor='black', rwidth=0.8)
        ax1.hist(bad_preds, bins=bins, alpha=0.5, label='Bad Hist', color='red', edgecolor='black', rwidth=0.8)
        
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
    
