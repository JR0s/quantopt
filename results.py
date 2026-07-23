import argparse
import time
from collections import defaultdict
import copy
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quapy as qp
from stopping_instanceSelection import RandomStop
from stopping_instanceSelection import RankingStop
from stopping_instanceSelection import EBGstop
from stopping_instanceSelection import WilcoxonStop
from stopping_instanceSelection import BaseSampling


from ast import literal_eval

# Method to load the specified file for a given quantifier
def unpack_data(file):
    data = pd.read_csv(file, index_col=0)
    data["p_est"] = data["p_est"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    data["p_val"] = data["p_val"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    # class_weight is saved as na but the hyperparameter has value None (string here, type None leads to errors)
    data["class_weight"] = data["class_weight"].fillna("None")
    data["class_weight"] = data["class_weight"].replace("<NA>", "None")
    #data["class_weight"] = data["class_weight"].where(data["class_weight"].notna(), "None")
    print(data["class_weight"].isna().sum())
    return data

def unpack_result(file):
    data = pd.read_csv(file, index_col=0, dtype={
        "strategy": "string",
        "fold_nr": "int",
        "error@100": "float",
        "quantifier@100": "string",
        "C@100": "float",
        "class_weight@100": "string",
        "n_evaluations" : "int"})
    data["class_weight@100"] = data["class_weight@100"].fillna("None")
    return data

# Method for computing the selected error on the data
def aggregate(data):
    averaged_best = data.groupby("strategy")[["error@100", "n_evaluations"]].mean().reset_index()

    maxs = (data.groupby("strategy", as_index=False)["n_evaluations"].max().rename(columns={"n_evaluations": "max_n"}))
    mins = (data.groupby("strategy", as_index=False)["n_evaluations"].min().rename(columns={"n_evaluations": "min_n"}))
    counts = (data.groupby("strategy").apply(
        lambda df: (df[["quantifier@100", "C@100", "class_weight@100"]].apply(tuple, axis=1).value_counts().to_dict())
    , include_groups=False).rename("config_counts").reset_index())

    averaged_best = averaged_best.merge(maxs, on="strategy")
    averaged_best = averaged_best.merge(mins, on="strategy")
    averaged_best = averaged_best.merge(counts, on="strategy")

    return averaged_best

def evaluate(data, results, error):

    agg_res = aggregate(results)
    print(agg_res)

    def compute_error(gdata):
        match error:
            case "mae":
                return qp.error.mae(gdata["p_est"], gdata["p_val"])
            case "mrae":
                return qp.error.mrae(gdata["p_est"], gdata["p_val"], eps=1/(2*gdata["val_sample"].nunique()))
            case "mkld":
                p_estimates = np.vstack(gdata["p_est"].to_numpy())
                p_vals = np.vstack(gdata["p_val"].to_numpy())
                return qp.error.mkld(p_estimates, p_vals, eps=1/(2*gdata["val_sample"].nunique()))

    # for saving the results
    file_time = time.localtime()
    file_time = str(file_time.tm_year) + "_" + str(file_time.tm_mon) + "_" + str(file_time.tm_mday) + "_" + str(file_time.tm_hour) + "_" + str(file_time.tm_min) + "_" + str(file_time.tm_sec)

    # calculate information about the dataset
    n_configurations = len(data[["quantifier", "C", "class_weight"]].drop_duplicates())
    val_samples = pd.unique(data["val_sample"])

   
   # Calculate the error of each configuration for the full data for calculating the differences with the correct error
    error_at_100 = data.groupby(["quantifier", "C", "class_weight"]).apply(compute_error, include_groups=False)
    error_at_100 = pd.DataFrame(error_at_100, columns=["error"]).reset_index()

    print(error_at_100.head(126))
    print(len(error_at_100))

    random_min_val = error_at_100["error"].min()

    # plot the diagrams for the random stopping
    random_strategies = ["10%random", "20%random", "30%random", "40%random", "50%random", "60%random", "70%random", "80%random", "90%random", "100%random"]
    x = np.linspace(0.1, 1, 10)
    mean_error = []
    min_error = []
    max_error = []

    for strat in random_strategies:
        mean = agg_res.loc[agg_res["strategy"] == strat, "error@100"].iloc[0]
        mean_error.append(mean)
        keys = list(agg_res.loc[agg_res["strategy"] == strat, "config_counts"].iloc[0].keys())
        items = list(agg_res.loc[agg_res["strategy"] == strat, "config_counts"].iloc[0].items())
        mask = error_at_100.apply(lambda row: (row["quantifier"], row["C"], row["class_weight"]) in keys, axis=1)
        min_error.append(abs(error_at_100.loc[mask, "error"].min()-mean))
        max_error.append(abs(error_at_100.loc[mask, "error"].max()-mean))

    plt.grid(alpha=0.4)
    plt.tight_layout(pad=4.0)
    plt.errorbar(x, mean_error, yerr=[min_error, max_error], fmt="o", color="red", ecolor="black", capsize=4)
    plt.axhline(y = random_min_val, color="gray", linestyle="--")
    plt.xlabel("Percentage of Data")
    plt.ylabel(error)
    plt.title("Additional Error\n of selecting other than the best configuration\n on each fold of instances")
    plt.savefig("plots_percentage_" + file_time + ".png")
    plt.close()

    #scatter plots for each method: each fold as datapoint with error and n_samples on the axis
    # for strategy in results["strategy"]:
    #     x = list(results[results["strategy"] == strategy]["n_evaluations"])
    #     y = list(results[results["strategy"] == strategy]["error@100"])
        
    #     plt.grid(alpha=0.2)
    #     plt.tight_layout(pad=4.0)
    #     plt.scatter(x, y, s=30,c="tab:red",marker="o", alpha=0.3)
    #     plt.xlabel("Number of evaluations")
    #     plt.ylabel("Real error @ 100")
    #     plt.title(f"Combination of the selected error\nfor each fold under n_evaluations\nunder {quantifier} of strategy {strategy}")
    #     plt.savefig("scatter_" + strategy + "_" + quantifier + "_" + file_time + ".png")
    #     plt.close()


    print("Print the scatterplots for the strategies...")
    for strategy in results["strategy"]:
        # count identical points
        counts = results[results["strategy"] == strategy].groupby(["n_evaluations", "error@100"]).size().reset_index(name="count")
        plt.grid(alpha=0.2)
        plt.tight_layout(pad=4.0)
        plt.xlabel("Number of evaluations")
        plt.ylabel("Real error @ 100")
        plt.scatter(counts["n_evaluations"],counts["error@100"],s=50,c=counts["count"], vmin=0, vmax=sum(counts["count"]), cmap="viridis")
        plt.colorbar(label="frequency")
        plt.xlim(0, len(val_samples))
        #plt.ylim(error_at_100["error"].min(), error_at_100["error"].max())
        plt.title(f"Combination of the selected error\nfor each fold under n evaluations\n of strategy {strategy}")
        plt.savefig("scatter_" + strategy + "_" + file_time + ".png")
        plt.close()



    def scatter(df, strat_name):
        plt.grid(alpha=0.4)
        plt.tight_layout(pad=4.0)
        plt.scatter(df["n_evaluations"], df["error@100"], s=30,c="tab:red",marker="o", alpha=0.7)
        plt.xlabel("Number of evaluations")
        plt.ylabel("Real error @ 100")
        plt.title(f"Combination of the selected error under n evaluations\n of strategy {strat_name}")
        plt.savefig("scatter_accumulated_" + strat_name + "_" + "_" + file_time + ".png")
        plt.close()
    
    # one scatter plot for all method types
    random_s = agg_res[agg_res["strategy"].str.endswith("random")]
    ranking_s = agg_res[agg_res["strategy"].str.startswith("top")]
    wilcoxon_s = agg_res[agg_res["strategy"].str.startswith("wilcoxon")]
    ebg_s = agg_res[agg_res["strategy"].str.startswith("EBG")]

    scatter(random_s, "RandomStop")
    scatter(ranking_s, "RankingStop")
    scatter(wilcoxon_s, "WilcoxonStop")
    scatter(ebg_s, "EBGStop")

    # one scatter plot for all best types
    smallest_random = random_s.loc[random_s["error@100"].idxmin()]
    smallest_ranking = ranking_s.loc[ranking_s["error@100"].idxmin()]
    smallest_wilcoxon = wilcoxon_s.loc[wilcoxon_s["error@100"].idxmin()]
    smallest_ebg = ebg_s.loc[ebg_s["error@100"].idxmin()]

    x_smallest = [smallest_random["n_evaluations"], smallest_ranking["n_evaluations"], smallest_wilcoxon["n_evaluations"], smallest_ebg["n_evaluations"]]
    y_smallest = [smallest_random["error@100"], smallest_ranking["error@100"], smallest_wilcoxon["error@100"], smallest_ebg["error@100"]]
    labels = ["RandomStop","RankingStop", "WilcoxonStop", "EBGStop"]

    colors = plt.cm.tab10.colors
    plt.grid(alpha=0.4)
    plt.tight_layout(pad=4.0)
    for i,(xi, yi, label) in enumerate(zip(x_smallest, y_smallest, labels)):
        plt.scatter(xi, yi, s=30,c=colors[i], cmap="viridis",marker="o", alpha=0.7, label=label)
    plt.xlabel("Number of evaluations")
    plt.ylabel("Real error @ 100")
    plt.title(f"Best performances for each strategy type")
    plt.legend()
    plt.savefig("scatter_best_" + "_" + file_time + ".png")
    plt.close()

# use file baseline_2026_7_17_9_49_49_lequa2022_T1B.csv as a test
# file baseline_2026_7_1_15_40_17_lequa2022_T1B.csv is whole set
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='results.py',
                    description='Evaluate the results and creation of plots',
                    epilog='see other resources')
    parser.add_argument("data_file", help="path to the directory of the saved data", type=str)
    parser.add_argument("-a", "--acc", help="path to the directory of the evaluation for acc", type=str)
    parser.add_argument("-p", "--pacc", help="path to the directory of the evaluation for pacc", type=str)
    parser.add_argument("-s", "--sld", help="path to the directory of the evaluation for sld", type=str)
    parser.add_argument("-e", "--error", help="used error metric", type=str)


    args = parser.parse_args()
    print("Running script:" + parser.prog)
    data_name = args.data_file
    result_acc = args.acc
    result_pacc = args.pacc
    result_sld = args.sld
    error = args.error

    res_acc = unpack_result(result_acc)
    res_pacc = unpack_result(result_pacc)
    res_sld = unpack_result(result_sld)
    results = pd.concat([res_acc, res_pacc, res_sld], ignore_index=True)

    stacked = pd.concat([res_acc, res_pacc, res_sld], keys=range(3))

    result = (stacked.groupby(level=1).apply(lambda g: g.loc[g["error@100"].idxmin()]).reset_index(drop=True))

    result.to_csv("merged.csv")
    data = unpack_data(data_name)
    evaluate(data, result, error)
