import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def get_value(y, index):
    x = y[index]
    return x


def diffs(probas):
    diffs = [
        abs(sum1 - sum2)
        for sum1, sum2 in [
            (sum(probas[: i + 1]), sum(probas[i:])) for i in range(1, len(probas))
        ]
    ]
    return diffs


def binary_search(commits):
    num_of_runs = 0
    len_commits = len(commits)
    low, high = 0, len_commits - 1

    state_of_first = get_value(commits, 0)
    state_of_last = get_value(commits, len_commits - 1)

    if state_of_first == 0 and state_of_last == 0:
        return -1, num_of_runs
    
    prev_mid = -1
    while low < high:
        print(f"num_of_runs: {num_of_runs}, low: {low}, high: {high}", end="")
        mid = (low + high) // 2

        num_of_runs += 1
        if commits[mid] == 1:
            high = mid

        elif commits[mid] == 0:
            low = mid + 1
        
        print(f", mid: {prev_mid}")
        prev_mid = mid

    return num_of_runs, high


def hybrid_search(commits, probabilities, alpha):
    num_of_runs = 0
    len_commits = len(commits)
    low, high = 0, len_commits - 1

    state_of_first = get_value(commits, 0)
    state_of_last = get_value(commits, len_commits - 1)

    if state_of_first == 0 and state_of_last == 0:
        print("HI!")
        return -1, num_of_runs

    while low < high:
        binary_mid = (low + high) // 2

        diffs_arr = diffs(probabilities[low : high + 1])

        prob_mid = np.argmin(diffs_arr) + low

        new_mid = int(((alpha * prob_mid) + ((1 - alpha) * binary_mid)))

        num_of_runs += 1
        if commits[new_mid] == 1:
            high = new_mid

        elif commits[new_mid] == 0:
            low = new_mid + 1

        print(f"run: {num_of_runs}, low: {low}, mid: {prob_mid}, high: {high}")

    return num_of_runs, high


# data generation
def get_preds_per_depth():
    samples = CONFIG["samples"]
    features = CONFIG["features"]
    n_informative = CONFIG["n_informative"]

    X, y = make_classification(
        n_samples=samples,
        n_features=features,
        n_redundant=10,
        n_informative=n_informative,
        n_clusters_per_class=1,
        n_classes=2,
        weights=[0.997, 0.003],
        flip_y=0.02,
        random_state=42,
    )

    data = {}
    for max_depth in tqdm(
        range(CONFIG["min_max_depth"], CONFIG["max_max_depth"] + 1), desc="Max Depth Progress", colour="green"
    ):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X, y)

        probas = model.predict_proba(X)[:, 1]
        data[max_depth] = {"preds": probas, "commits": y}

    return data


def transform_data(y, predictions):
    min_chunk_size = 2 ** CONFIG["min_log_size"]
    max_chunk_size = 2 ** CONFIG["max_log_size"]

    index = 0
    chunks = []
    while True:
        size = np.random.randint(min_chunk_size, max_chunk_size)

        if index + size >= len(y):
            break

        y_chunk = y[index : index + size]
        preds_chunk = predictions[index : index + size]

        if sum(y_chunk) == 0:
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
    print("Starting Hybrid Search")
    data = get_preds_per_depth()
    print("Data Generation Complete")
    alpha = 0.5

    for max_depth, data_entries in data.items():
        preds = data_entries["preds"]
        commits = data_entries["commits"]
        chunks = transform_data(commits, preds)
        break

    trials = 0
    for chunk in chunks:
        y_chunk = chunk["y_chunk"]
        preds_chunk = chunk["preds_chunk"]
        y_transformed = chunk["y_transformed"]

        print(y_transformed)
        print()

        # hybrid search
        print("Binary Search")
        hybrid_runs, hybrid_index = hybrid_search(y_transformed, preds_chunk, alpha)

        # compare with alpha = 0
        print("\nHeruistic Search")
        binary_runs, binary_index = binary_search(y_transformed)

        # assert they are the same index
        assert binary_index == hybrid_index
        print(f"saved runs: {(binary_runs - hybrid_runs)/(binary_runs - 2)}")
        trials += 1
        print("\n\n")
        if trials == 2:
            break



CONFIG = {
    "samples": 40000,
    "features": 100,
    "n_informative": 50,
    "min_max_depth": 50,
    "max_max_depth": 50,
    "min_log_size": 4,
    "max_log_size": 7,
}


main()
