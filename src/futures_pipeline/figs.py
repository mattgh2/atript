import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import dominate 
from dominate.tags import div, img, body, h1, link, p
from typing import cast
from .config import PROJECT_ROOT
from datetime import datetime

def plot_predictions(y_pred, y_true, save_to: Path | None = None):
    days = np.arange(1, len(y_pred) + 1)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(days, y_pred, label="Predicted", color="black")
    ax.plot(days, y_true, label="Actual", color="red")

    ax.set(
            title="Actual vs. Predicted",
            xlabel="days",
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

@param fig_path The pathname for the directory of figs.
@param page_path the pathname where the html file is located.
@param text Optional text to display alongside a figure. 
"""
def display_figs(fig_path: Path, page_path: Path, text: dict[str, list] | None = None):
    last_update = datetime.today()
    doc: dominate.document = dominate.document(title="foo")
    with doc.head:
        link(rel="stylesheet", href="src/output.css")
    with doc:
        with div(f"last_update: {str(last_update)}"):
            ...
        with div(id='images', cls="flex flex-wrap"):
            for fig in fig_path.iterdir():
                if fig.suffix == ".png":
                    with div(id=f"{fig.stem}-cnt", cls="flex flex-col items-center"):
                        with div(id=fig.stem):
                            img(src=fig.resolve())
                        if text and text.get(fig.stem):
                            with div(id=f"{fig.name}-text"):
                                for item in text[fig.stem]:
                                    p(item)

                            


    with open(page_path, 'w') as f:
        f.write(str(doc))

