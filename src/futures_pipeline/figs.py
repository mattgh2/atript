import matplotlib.pyplot as plt


def plot_predictions(days, y_pred, y_true):
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(days, y_pred, label="Predicted", color="black")
    ax.plot(days, y_true, label="Actual", color="red")

    ax.set(
            title="Actual vs. Predicted",
            xlabel="days",
            ylabel="price"
    )

    ax.legend()
    
    plt.show()


