import json

from scipy.interpolate import Akima1DInterpolator
import numpy as np

from Code.DistributionGenerator.parse_dict import fix_keys


def bin_dict(
    d: dict,
    /,
    n_bins: int = 100,
    *,
    func: callable = np.sum,
    range_values: tuple = None,
    single_key: bool = False,
) -> dict:
    """Bins a dictionary of values into n_bins bins and applies a function to each bin.

    Args:
        d (dict): Dictionary of values to bin.
        n_bins (int, optional): Number of bins. Defaults to 100.
        func (callable, optional): Function to apply to each bin. Defaults to np.sum. Should be a numpy function, or any function that returns an object that has a .item() method that returns a primitive float/int.
        range_values (tuple, optional): Tuple of min and max values to consider. Defaults to None. If None, the min and max values of the dictionary keys are used.
        single_key (bool, optional): Whether to return a dictionary with a single key. Defaults to False.

    Returns:
        dict: Dictionary with the binned values.
    """
    if range_values is None:
        range_values = min(d.keys()), max(d.keys())

    bins = np.linspace(
        range_values[0], range_values[1], n_bins + 1
        # convert this to a regular list instead of a numpy array, better for later computations
    ).tolist()

    binned_data = {}

    for i in range(len(bins) - 1):
        binned_data[(bins[i], bins[i + 1])] = func(
            [v for k, v in d.items() if bins[i] <= k < bins[i + 1]]
            # .item() is to ensure the output is a primitive float/int and not a numpy float/int
        ).item()

    if single_key:
        ret = {}
        for k, v in binned_data.items():
            ret[k[0]] = v

        return ret

    # type_conv_dict = {
    #     np.float64: float,
    #     np.int64: int
    # }

    # ret = {}
    # for k, v in binned_data.items():
    #     new_k = type_conv_dict[type(k)](k)
    #     new_v = type_conv_dict[type(v)](v)
    #     ret[new_k] = new_v.value

    return binned_data


# Some safe functions to use with bin_dict
def _mean(x: list) -> np.number:
    return np.mean(x) if len(x) > 0 else np.float64(0)


def _sum(x: list) -> np.number:
    return np.sum(x) if len(x) > 0 else np.float64(0)


def interpolate_missing_parts(to_bin: dict, /, *, num_points: int = 1000, plot: bool = False, method: str = "makima") -> dict:
    """Interpolates data using the Akima method.

    Args:
        to_bin (dict): Dictionary to interpolate.
        num_points (int, optional): Number of points to interpolate. Defaults to 1000. This is the number of points in the final interpolated data.
        plot (bool, optional): Whether to plot the data. Defaults to False.
        method (str, optional): Interpolation method. Defaults to "makima". Makima is the Modified Akima interpolation method and performs better and more smoothly.

    Returns:
        dict: Interpolated dictionary. Size is num_points.

    Example:
        >>> to_bin = {0.1: 10, 0.2: 20, 0.3: 30}
        >>> interpolated_dict = interpolate_missing_parts(to_bin, num_points=5, method="makima")
        >>> print(interpolated_dict)
        {0.1: 10.0, 0.15: 15, 0.2: 20.0, 0.25: 25.0, 0.3: 30.0}
    """
    # Get the keys and values of the dictionary
    x = np.array(list(to_bin.keys()))
    y = np.array(list(to_bin.values()))

    # Get the points to interpolate
    # convert to a list of primitive ints
    x_new = np.linspace(min(x), max(x), num_points).tolist()

    # Interpolate the data, this uses the Akima method with the specified variant from the keyword arguments specified
    y_new = Akima1DInterpolator(x, y, method=method)(x_new).tolist()
    interpolated_dict = dict(zip(x_new, y_new))
    return interpolated_dict


def main():
    pass


# if __name__ == "__main__":