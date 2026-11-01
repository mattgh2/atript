import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import dominate 
from dominate.tags import div, img, body, h1, link, p, a
from typing import Optional
from .config import PROJECT_ROOT
from datetime import datetime
from collections.abc import Iterable

def plot_predictions(y_pred, y_true, lower_p, upper_p, save_to: Path | None = None):
    candles = np.arange(1, len(y_pred) + 1)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(candles, y_pred, label="Predicted", color="black")
    ax.plot(candles, y_true, label="Actual", color="red")

    ax.fill_between(
            candles,
            lower_p,
            upper_p,
            alpha=0.7,
            label="Prediction interval",
            color="xkcd:light lavender"
    )

    ax.set(
            title="Actual vs. Predicted",
            xlabel="candles",
            ylabel="price"
    )

    ax.legend()

    if save_to:
        plt.savefig(save_to)
        plt.close(fig)
    else:
        plt.show()


def parameter_performance(
    param_name, 
    metric_name, 
    param_values, 
    scores, 
    save_to: Path | None = None
):
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(param_values, scores)
    ax.set(xlabel=param_name, ylabel=metric_name)

    if save_to:
        fig.savefig(save_to)
        plt.close(fig)
    else:
        plt.show()

