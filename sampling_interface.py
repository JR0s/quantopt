import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quapy as qp

# Method to load the specified file for a given quantifier
def unpack(file, quantifier):
    df = pd.read_csv(file, index_col=0)
    data = df.loc[df["quantifier"] == quantifier]
    data["p_est"] = data["p_est"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    data["p_val"] = data["p_val"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    # is saved as na but the hyperparameter has value None (string here, type None leads to errors)
    data["class_weight"] = data["class_weight"].where(data["class_weight"].notna(), "None")
    return data

# Method for plotting the additional error of selecting different configurations at lower percentages than the best performing configuration with 100% of data
def plot(data, error, folds, quantifier):
    file_time = time.localtime()
    file_time = str(file_time.tm_year) + "_" + str(file_time.tm_mon) + "_" + str(file_time.tm_mday) + "_" + str(file_time.tm_hour) + "_" + str(file_time.tm_min) + "_" + str(file_time.tm_sec)
    filename = "percentage_sampling_" + quantifier + "_" + file_time + "_" + ".csv"

    result = []

    # Local method for computing the selected error on the data
    def compute_error(gdata):
        match error:
            case "mae":
                return qp.error.mae(gdata["p_est"], gdata["p_val"])
            case "mrae":
                return qp.error.mrae(gdata["p_est"], gdata["p_val"])
            case "mkld":
                return qp.error.mkld(gdata["p_est"], gdata["p_val"])
            
    # Calculate the error of each configuration for the full data for calculating the differences with the correct error
    error_at_100 = data.groupby(["C", "class_weight"]).apply(compute_error, include_groups=False)
    error_at_100 = pd.DataFrame(error_at_100, columns=["error"]).reset_index()

    percentages = np.linspace(0.1, 1, 10) #hier stopping+instance selection schnittstelle
    # Calculate the performance value and index for each configuration on each fold of each fraction of data
    best_performance = []
    for i, percentage in enumerate(percentages): #hier stopping+instance selection schnittstelle
        rand = np.random.RandomState(seed=42)
        acc = defaultdict(list)
        for f in range(folds):
            events = data.sample(frac=percentage, random_state = rand) #hier stopping+instance selection schnittstelle
            events = events.groupby(["C", "class_weight"]).apply(compute_error, include_groups=False)
            events = pd.DataFrame(events, columns=["error"]).reset_index()
            
            for k, v in events.items():
                acc[k].append(v)

            # for second plot
            best_performance.append({
                "C": events.iloc[np.argmin(events["error"])]["C"],
                "class_weight": events.iloc[np.argmin(events["error"])]["class_weight"],
                "min_value": min(events["error"]),
                "min_index": np.argmin(events["error"]), # argmin (= "beste" config im fold) -> error_at_100[argmin]
                "fold_id": f,
                "percentage": percentage
            })
        
    best_performance = pd.DataFrame(best_performance)

    variance = np.empty(len(percentages))
    std_dev = np.empty(len(percentages))
    mean = np.empty(len(percentages))
    for i, perc in enumerate(percentages):
        values = best_performance.loc[best_performance["percentage"] == perc, ["min_index", "C", "class_weight"]]
        
        errors = pd.merge(values, error_at_100, on=["C", "class_weight"])["error"]
        mean[i] = np.mean(errors)
        variance[i] = np.var(errors)
        std_dev[i] = np.std(errors)

    # Plot the graph
    plt.grid(alpha=0.4)
    plt.tight_layout(pad=4.0)
    plt.errorbar(percentages, mean, yerr=std_dev, fmt="o", color="red", ecolor="black", capsize=4)
    plt.axhline(y = mean[len(mean)-1], color="gray", linestyle="--")
    plt.xlabel("Percentage of Data")
    plt.ylabel(error)
    plt.title("Additional Error\n of selecting other than the best configuration\n on each fold of instances for quantifier " + quantifier)
    plt.savefig("plots_percentage_" + quantifier + "_" + file_time + ".pdf")
    plt.close()

    return result


# use file baseline_2026_6_29_11_11_21_lequa2022_T1B.csv first
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='percentual_sampling.py',
                    description='Create Plots looking at the performance for percentual sampling',
                    epilog='see other resources')
    parser.add_argument("filename", help="path to the directory of the saved data", type=str)
    parser.add_argument("-e", "--error_metric", help="error metric to be looked at", type=str)
    parser.add_argument("-f", "--folds", help="number of folds for calculating mean and variance", type=int)
    parser.add_argument("--test", help="toggle test mode", action="store_true")

    args = parser.parse_args()
    print("Running script:" + parser.prog)
    file = args.filename
    error = args.error_metric
    folds = args.folds
    
    quantifier = ["ACC", "PACC", "SLD"]
    for q in quantifier:
        data = unpack(file, q)
        plot(data, error, folds, q)

