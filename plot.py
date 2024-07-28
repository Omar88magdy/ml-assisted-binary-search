import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
from tqdm import tqdm


def plot(
    algorithm,
    metric=None,
    *,
    samples=15000,
    scores=None,
    saved_runs=None,
    save_fig=False,
    transparent=False,
    n_informative=-1,
    **kwargs,
):

    saved_runs_means = saved_runs
    scores_means = scores
    plt.figure(figsize=(18, 9))

    # Find the vertical line where saved runs become negative
    negative_indices = [
        i for i, saved_run in enumerate(saved_runs_means) if saved_run < 0
    ]

    if negative_indices:
        negative_index = negative_indices[
            -1
        ]
        x1 = scores_means[negative_index]
        y1 = saved_runs_means[negative_index]
        try:
            x2 = scores_means[negative_index + 1]
            y2 = saved_runs_means[negative_index + 1]
        except IndexError:
            x2 = scores_means[negative_index - 1]
            y2 = saved_runs_means[negative_index - 1]

        x_zero_cross = x1 + (0 - y1) / (y2 - y1) * (x2 - x1)

        plt.axvline(
            x=x_zero_cross,
            color="#EC6602",
            linestyle="--",
            linewidth=1.5,
            label="_nolegend_",
        )
        plt.axhline(
            y=0,
            color="#EC6602",
            linestyle="--",
            linewidth=1.5,
            label="_nolegend_",
        )
        x_zero_cross = f"{x_zero_cross:.4f}"
        plt.axhline(y=0, color="#EC6602", linestyle="--", linewidth=1.5)

    else:
        x_zero_cross = None

    plt.legend(
        [f"Samples: {samples}\nThreshold: {x_zero_cross}"],
        loc="upper right",
        fontsize=10,
        frameon=False,
        shadow=False,
        title="Info",
        title_fontsize="20",
    )

    if metric == "Weighted Binary Crossentropy Loss":
        plt.xlabel(f"Loss Values ({metric})", fontsize=25)
    else:
        plt.xlabel(f"Score Values ({metric})", fontsize=25)
    plt.ylabel("Saved Runs", fontsize=25)

    ax = plt.gca()

    plt.grid(True)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=25))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=13))

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Remove borders
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # plt.ylim(-16, 0.90)
    # plt.xlim(0, 1)
    plt.plot(
        scores_means,
        saved_runs_means,
        marker="o",
        linestyle="--",
        color="#009999",
        label="_nolegend_",
        linewidth=2,
        **kwargs,
    )

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    if save_fig:
        # if transparent:
        #     transparency = "transparent"
        # else:
        #     transparency = "non-transparent"

        directory = "Graphs/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(
            os.path.join(
                directory,
                f"{algorithm}_{metric}_{n_informative}.png",
            ),
            transparent=transparent,
            dpi=100,
        )

    plt.close();


def parse_file_name(file_name):
    file_name = file_name.split("__")
    algorithm = file_name[0]
    samples = int(file_name[1].split("_")[1])
    features = int(file_name[2].split("_")[1])
    n_informative = int(file_name[3].split("_")[2])
    min_max_depth = int(file_name[4].split("_")[3])
    max_max_depth = int(file_name[5].split("_")[3].split(".")[0])

    return (algorithm, samples, features, n_informative, min_max_depth, max_max_depth)


def main():
    # list all files in results
    files = os.listdir("results")

    for file in tqdm(files):
        algorithm, samples, _, n_informative, _, _ = parse_file_name(file)

        df: pd.DataFrame = pd.read_pickle(f"results/{file}")

        df = pd.concat(
            [df.drop(["score"], axis=1), df["score"].apply(pd.Series)], axis=1
        )

        df["mean_saved_runs"] = df["saved_runs"].map(lambda x: np.mean(x))

        metrics = ["b3s", "recall", "tnr", "f1", "htp", "wbce", "focal"]

        for metric in metrics:
            df.sort_values(by=metric, inplace=True)
            plot(
                algorithm=algorithm,
                metric=metric,
                samples=samples,
                scores=list(df[metric]),
                saved_runs=list(df["mean_saved_runs"]),
                save_fig=True,
                transparent=True,
                markersize=9,
                n_informative=n_informative,
            )


if __name__ == "__main__":
    main()