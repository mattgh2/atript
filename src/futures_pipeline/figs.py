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


"""
Writes a directory of figures to html.

@param fig_path
@param page_path the pathname where the html file is located.
"""
def display_figs(figs: list[tuple[Path, list[str]]], page_path: Path):
    page_path.touch(exist_ok=True)
    last_update = datetime.today()
    doc: dominate.document = dominate.document(title="foo")
    with doc.head:
        link(rel="stylesheet", href="../../src/output.css")
    with doc:
        with div():
            with a("Back", href="../../../main.html"): ...
        with div(f"last_update: {str(last_update)}"): ...
        with div(id='images', cls="flex flex-wrap"):
            for fig, text in figs:
                with div(id=f"{fig.stem}-cnt", cls="flex flex-col items-center"):
                    with div(id=fig.stem):
                        img(src=fig.resolve())
                    if len(text):
                        with div(id=f"{fig.name}-text"):
                            for item in text:
                                p(item)


    with open(page_path, 'w') as f:
        f.write(str(doc))

