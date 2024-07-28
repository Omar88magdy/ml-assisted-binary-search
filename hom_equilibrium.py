import os
import math
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


def diffs(probas):
    left_sums = np.cumsum(probas)[:-1]
    right_sums = np.sum(probas) - left_sums
    return np.abs(left_sums - right_sums)


def equilibrium(data_transformed, probas):
    num_of_runs = 0

    data_transformed_len = len(data_transformed)
    low, high = 0, data_transformed_len - 1

    state_of_first = get_value(data_transformed, 0)
    state_of_last = get_value(data_transformed, data_transformed_len - 1)

    if state_of_first == 0 and state_of_last == 0:
        return -1, num_of_runs

    heuristic_runs = []

    previous_mid = -1
    while len(probas[low:high]) >= 1:
        diffs_arr = diffs(probas[low : high + 1])

        mid = np.argmin(diffs_arr) + low

        state_of_current = get_value(data_transformed, mid)
        num_of_runs += 1

        if state_of_current == 0 and previous_mid != mid:
            low = mid + 1

        elif state_of_current == 1 and previous_mid != mid:
            high = mid

        else:
            return num_of_runs, high

        previous_mid = mid

    heuristic_runs.append(num_of_runs)
    return num_of_runs, high


def wBCE_loss(y, y_hat):
    # Calculate the lambda value
    lam = (len(y) - np.sum(y)) / len(y)

    # Ensure y_hat values are in (0, 1)
    epsilon = 1e-8
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    # Calculate the weighted BCE loss
    wBCE_loss = (-2 / len(y)) * np.sum(
        lam * y * np.log(y_hat) + (1 - lam) * (1 - y) * np.log(1 - y_hat)
    )

    return wBCE_loss


def focal_loss(y, y_hat, gamma=2):
    # Calculate the lambda value
    lam = (len(y) - np.sum(y)) / len(y)

    # Ensure y_hat values are in (0, 1)
    epsilon = 1e-8
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    # Calculate the focal loss
    focal_loss = -np.mean(
        lam * y * (1 - y_hat) ** gamma * np.log(y_hat)
        + (1 - lam) * (1 - y) * y_hat ** gamma * np.log(1 - y_hat)
    )

    return focal_loss


def hmean(x, y):
    return 2 * (x * y) / (x + y + CONFIG["epsilon"])


def scoring(*, y_true, y_pred=None, metric="b3s", is_thresholded=False):
    """
    input: true_labels, predicted_labels, predicted_probabilities, metric
    output: score
    metric: 'b3s', 'htp', 'f1', 'precision', 'recall', 'tnr', 'wbce'
    """
    if metric != "wbce":
        if is_thresholded:
            predictions = y_pred
        else:
            predictions = np.where(y_pred >= CONFIG["threshold"], 1, 0)

        cm = confusion_matrix(y_true, predictions)
        tn, fp, fn, tp = cm.ravel()

    # Define available metrics
    if metric == "b3s":
        score = hmean(
            scoring(
                y_true=y_true, y_pred=predictions, metric="recall", is_thresholded=True
            ),
            scoring(
                y_true=y_true, y_pred=predictions, metric="tnr", is_thresholded=True
            ),
        )
    elif metric == "htp":
        score = hmean(
            scoring(
                y_true=y_true, y_pred=predictions, metric="tnr", is_thresholded=True
            ),
            scoring(
                y_true=y_true,
                y_pred=predictions,
                metric="precision",
                is_thresholded=True,
            ),
        )
    elif metric == "f1":
        score = hmean(
            scoring(
                y_true=y_true,
                y_pred=predictions,
                metric="precision",
                is_thresholded=True,
            ),
            scoring(
                y_true=y_true, y_pred=predictions, metric="recall", is_thresholded=True
            ),
        )
    elif metric == "precision":
        score = (tp) / (tp + fp) if (tp + fp) != 0 else 0.0
    elif metric == "recall":
        score = (tp) / (tp + fn) if (tp + fn) != 0 else 0.0
    elif metric == "tnr":
        score = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    elif metric == "wbce":
        if y_pred is None:
            raise ValueError("y_pred_proba must be provided for wbce")
        score = wBCE_loss(y_true, y_pred)
    elif metric == "focal":
        score = focal_loss(y_true, y_pred)
    else:
        raise ValueError(f"Metric '{metric}' not recognized")

    return score


def add_uniform_noise(probas, noise_level=0.1):
    """
    Adds random uniform noise to a list of probabilities.

    Parameters:
    probas (list of float): List of probabilities to add noise to.
    noise_level (float): Maximum amount of noise to add. Noise will be in the range [-noise_level, noise_level].

    Returns:
    list of float: Probabilities with added noise.
    """
    noise = np.random.uniform(-noise_level, noise_level, len(probas))
    probas += noise
    probas = (probas - min(probas)) / (max(probas) - min(probas))

    return probas


def get_value(y, index):
    x = y[index]
    return x


def binary_search(commits):
    num_of_runs = 0
    len_commits = len(commits)
    low, high = 0, len_commits - 1

    state_of_first = get_value(commits, 0)
    state_of_last = get_value(commits, len_commits - 1)

    if state_of_first == 0 and state_of_last == 0:
        return -1, num_of_runs

    if CONFIG["print_detailed_search"]:
        print(f"low: {low}, high: {high}")
    prev_mid = -1
    while low < high:
        mid = (low + high) // 2

        num_of_runs += 1
        if commits[mid] == 1:
            high = mid

        elif commits[mid] == 0:
            low = mid + 1

        if CONFIG["print_detailed_search"]:
            print(f"runs: {num_of_runs}, low: {low}, mid: {prev_mid}, high: {high}")
        prev_mid = mid

    return num_of_runs, high


def hit_or_miss(y_transformed, probas):
    heuristic_runs = 0
    # create an array of -1s to keep track of the dynamic programming, use numpy for speed
    dynamic = np.full(len(y_transformed), -1)

    # enumerate the probabilities and their indexes, use numpy for speed
    top_idxs = np.argsort(probas)[::-1]

    for idx in top_idxs:
        if dynamic[idx] != -1:
            continue

        test_outcome = get_value(y_transformed, idx)
        heuristic_runs += 1

        dynamic[idx] = test_outcome

        if test_outcome == 1:
            prev_outcome = get_value(y_transformed, idx - 1)
            heuristic_runs += 1
            dynamic[idx - 1] = prev_outcome

            if prev_outcome == 0:
                return heuristic_runs, idx

        first_fails = np.where((dynamic[:-1] == 0) & (dynamic[1:] == 1))[0]
        if len(first_fails) > 0:
            return heuristic_runs, first_fails[0] + 1

    raise ValueError("No hit or miss found")


def get_preds_per_depth(samples, features, n_informative, min_max_depth, max_max_depth):
    name = f"data__samples_{samples}__features_{features}__n_informative_{n_informative}__min_max_depth_{min_max_depth}__max_max_depth_{max_max_depth}.pkl"

    # check if the data is already generated
    try:
        data = pd.read_pickle(os.path.join("data", name))
        return data
    except:
        pass

    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_redundant=10,
        n_informative=n_informative,
        n_clusters_per_class=1,
        n_classes=2,
        weights=[0.997, 0.003],
        flip_y=0,
        random_state=42,
    )

    data = {}
    for max_depth in tqdm(
        range(CONFIG["min_max_depth"], CONFIG["max_max_depth"] + 1),
        desc="Max Depth Progress",
        colour="green",
    ):
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=8,
        )

        model.fit(X, y)

        probas = model.predict_proba(X)[:, 1]
        data[max_depth] = {"preds": probas, "commits": y}

    pd.DataFrame(data).to_pickle(os.path.join("data", name))

    return data


def transform_data(y, predictions):
    min_chunk_size = 2 ** (CONFIG["min_log_size"] - 1) + 1
    max_chunk_size = 2 ** CONFIG["max_log_size"]
    if CONFIG["print_detailed_search"]:
        print(f"min_chunk_size: {min_chunk_size}, max_chunk_size: {max_chunk_size}")

    index = 0
    chunks = []
    np.random.seed(42)
    while True:
        size = np.random.randint(min_chunk_size, max_chunk_size)

        if index + size >= len(y):
            break

        y_chunk = y[index : index + size]
        preds_chunk = predictions[index : index + size]

        if sum(y_chunk) == 0 or y_chunk[0] == 1:
            # Do not append!
            index += size
        elif sum(y_chunk) == 1:
            # Safely append
            index += size
            chunks.append({"y_chunk": y_chunk, "preds_chunk": preds_chunk})
        else:
            one_indices = [i for i, x in enumerate(y_chunk) if x == 1]

            # Make n_bad deep copies of the chunk
            y_copied_chunks = []
            pred_copied_chunks = []
            for one_index in one_indices:
                curr_chunk = y_chunk.copy()
                # set the current one_index to 1, and the rest to 0
                curr_chunk[:one_index] = 0
                curr_chunk[one_index + 1 :] = 0
                y_copied_chunks.append(curr_chunk)
                pred_copied_chunks.append(preds_chunk)

            # Append the copied chunks
            for y_copied_chunk, pred_copied_chunk in zip(
                y_copied_chunks, pred_copied_chunks
            ):
                chunks.append(
                    {"y_chunk": y_copied_chunk, "preds_chunk": pred_copied_chunk}
                )

            index += size

    for chunk in chunks:
        y_chunk = chunk["y_chunk"]
        preds_chunk = chunk["preds_chunk"]
        # Find the index of the single 1 in the chunk
        y_transformed = y_chunk.copy()
        y_transformed[np.argmax(y_chunk) + 1 :] = 1
        chunk["y_transformed"] = y_transformed
        assert len(y_chunk) == len(preds_chunk)
        assert sum(y_chunk) == 1

    return chunks


def main():
    data = get_preds_per_depth(
        CONFIG["samples"],
        CONFIG["features"],
        CONFIG["n_informative"],
        CONFIG["min_max_depth"],
        CONFIG["max_max_depth"],
    )

    results_dict = {}
    for max_depth, preds_commits in tqdm(
        data.items(),
        desc=f"Processing chunks",
        colour="blue",
    ):
        preds = preds_commits["preds"]
        commits = preds_commits["commits"]

        chunks = transform_data(commits, preds)

        y_true_concat = np.concat([chunk["y_chunk"] for chunk in chunks])
        y_pred_concat = np.concat([chunk["preds_chunk"] for chunk in chunks])

        scores = {
            k: scoring(y_true=y_true_concat, y_pred=y_pred_concat, metric=k)
            for k in CONFIG["metrics"]
        }

        results_dict[max_depth] = {
            "binary_runs": [],
            "algorithm_runs": [],
            "saved_runs": [],
            "score": scores,
        }

        for chunk in chunks:
            y_transformed = chunk["y_transformed"]

            # binary runs can be calculated only using the length of the array

            binary_runs, binary_index = binary_search(y_transformed)
            # binary_runs = np.ceil(np.log(len(y_transformed)))
            if CONFIG["print_detailed_search"]:
                print("---" * 20)

            # noise_level =(10*np.exp(-0.4*max_depth))/10

            # chunk["preds_chunk"] = add_uniform_noise(chunk["preds_chunk"], noise_level=noise_level)
            if CONFIG["algorithm"] == "equilibrium":
                heuristic_runs, heuristic_index = equilibrium(
                    y_transformed, chunk["preds_chunk"]
                )
            elif CONFIG["algorithm"] == "hom":
                heuristic_runs, heuristic_index = hit_or_miss(
                    y_transformed, chunk["preds_chunk"]
                )
            # check index similarity
            assert (
                binary_index == heuristic_index
            ), f"{binary_index=} != {heuristic_index=}, in " + " ".join(
                [
                    str((int(i), round(x, 4)))
                    for i, x in enumerate(y_transformed.tolist())
                ]
            )

            if CONFIG["print_detailed_search"]:
                print(f"Size: {np.ceil(np.log(len(y_transformed)))}")
                print(f"saved runs: {(binary_runs - heuristic_runs)/(binary_runs - 2)}")
                print("*" * 100 + "\n\n")

            results_dict[max_depth]["binary_runs"].append(binary_runs)
            results_dict[max_depth]["algorithm_runs"].append(heuristic_runs)
            results_dict[max_depth]["saved_runs"].append(
                (binary_runs - heuristic_runs) / (binary_runs - 2)
            )

    out_result_path = f"{CONFIG['algorithm']}__samples_{CONFIG['samples']}__features_{CONFIG['features']}__n_informative_{CONFIG['n_informative']}__min_max_depth_{CONFIG['min_max_depth']}__max_max_depth_{CONFIG['max_max_depth']}.pkl"
    pd.DataFrame(results_dict).T.to_pickle(os.path.join("results", out_result_path))


CONFIG = {
    "algorithm": "hom",  # equilibrium or hom
    "metrics": [
        "b3s",
        "htp",
        "f1",
        "wbce",
        "recall",
        "tnr",
        "focal",
    ],  # b3s, htp, f1, precision, recall, tnr, wbce
    "samples": 100000,
    "features": 100,
    "n_informative": 40,
    "min_max_depth": 1,
    "max_max_depth": 40,
    "min_log_size": 8,
    "max_log_size": 10,
    "print_detailed_search": False,
    "threshold": 0.5,
    "epsilon": 1e-8,
}

# parallelize get_preds_per_depth function
n_informatives = list(range(10, 100, 10))

for n_informative in tqdm(n_informatives):
    CONFIG["n_informative"] = n_informative
    main()