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
from stopping_instanceSelection import BaseSampling


# Method to load the specified file for a given quantifier
def unpack(file, quantifier):
    df = pd.read_csv(file, index_col=0)
    data = df.loc[df["quantifier"] == quantifier]
    data["p_est"] = data["p_est"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    data["p_val"] = data["p_val"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    # class_weight is saved as na but the hyperparameter has value None (string here, type None leads to errors)
    data["class_weight"] = data["class_weight"].where(data["class_weight"].notna(), "None")
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

# Main method for running the experiment
def experiment(data, error, folds, quantifier, n_jobs, batch_size_factor = 0.01, test_flag=False):
    # for saving the results
    file_time = time.localtime()
    file_time = str(file_time.tm_year) + "_" + str(file_time.tm_mon) + "_" + str(file_time.tm_mday) + "_" + str(file_time.tm_hour) + "_" + str(file_time.tm_min) + "_" + str(file_time.tm_sec)
    filename = "stopping_experiment_" + quantifier + "_" + error + "_" + file_time + "_"

    # Calculate the number of configurations and get the name/index of the validation samples
    n_configurations = len(data[["C", "class_weight"]].drop_duplicates())
    val_samples = pd.unique(data["val_sample"])

    # Set the number of accepted samples per evaluation step
    batch_size = int(batch_size_factor*len(val_samples)) # default: 1% of data
    if(test_flag):
        batch_size = 1
    print(f"The dataset has {len(val_samples)} many validation samples and a batchsize of {batch_size}")

    result = []

    # Local method for computing the selected error on the data
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
            
    # Calculate the error of each configuration for the full data for calculating the differences with the correct error
    error_at_100 = data.groupby(["quantifier", "C", "class_weight"]).apply(compute_error, include_groups=False)
    error_at_100 = pd.DataFrame(error_at_100, columns=["error"]).reset_index()

    # instantiate all strategies for early stopping
    stopping_strategies = {} # map from strategy name (key) to a Stopping object (value)
    percentages = np.linspace(0.1, 1, 10)
    for percentage in percentages:
        strategy_name = f"{int(percentage*100)}%random"
        stopping_strategies[strategy_name] = RandomStop(
            ["quantifier", "C", "class_weight"],
            int(percentage * len(val_samples)),
        )

    
    # Calculate the performance value and index for each configuration on each fold of each fraction of data
    best_performance = []
    worst_performance = []
    # we want to look at folds per stopping per configuration

    # function for evaluating the stopping on a batch of samples
    def eval_step(data, samples, strategy):
        dataset = data.copy()
        dataset.loc[dataset["val_sample"].isin(samples), "accepted"] = True # add whole batch to samples that are considered in evaluation 
        stopped = strategy(dataset[dataset["accepted"]]) # evaluate stopping on these samples

        # add stopping data to the dataset and clean it up
        dataset = dataset.merge(stopped.drop(columns=["p_est", "p_val", "t_est", "t_train", "accepted"]), on=("quantifier", "C", "class_weight", "val_sample"), how="left", validate="one_to_one")
        dataset["stopped_x"] = dataset["stopped_x"].astype("boolean")
        dataset["stopped_y"] = dataset["stopped_y"].astype("boolean")
        dataset["stopped"] = dataset["stopped_y"].combine_first(dataset["stopped_x"])
        dataset = dataset.drop(columns=["stopped_x", "stopped_y"])
        return dataset
    
    def eval_strategy(data, error, folds, quantifier, val_samples, batch_size, strategy_name, strategy):
        print(f"This is stopping strategy {strategy_name}")
        rng = np.random.default_rng(42)
        best_performance_one_run = []
        worst_performance_one_run = []
        for i in range(folds):
            # create a new stopping strategy for each fold such that the object params are fresh for every fold
            fold_strategy = copy.deepcopy(strategy)
            
            # copy the data so that it can be split without breaking the original object
            strategy_data = data.copy()

            # keep track of which evaluations are accepted by the strategy
            strategy_data["accepted"] = False
            strategy_data["stopped"] = False

            # initialize sampler on initialized state
            sampler = BaseSampling(val_samples, batch_size, rng=rng, starting_index=0)

            # accept the first N evaluations to initialize the strategy
            initial_samples = sampler.sampling()
            #print(f"init samples = {initial_samples}, batch_size = {batch_size}")
            strategy_data = eval_step(strategy_data, initial_samples, fold_strategy)

            # evaluate until all configurations have stopped
            while(not(strategy_data.groupby(["quantifier", "C", "class_weight"])["stopped"].any().all()) and sampler.iter < sampler.length):
                iteration_samples = sampler.sampling()
                strategy_data = eval_step(strategy_data, iteration_samples, fold_strategy)

            # among all accepted evaluations, compute the apparent error
            event = strategy_data[strategy_data["accepted"]].groupby(["quantifier", "C", "class_weight"]).apply(compute_error, include_groups=False)
            event = pd.DataFrame(event, columns=["error"]).reset_index()
        
            # find the best configuration according to the apparent error
            min_error = event.loc[event["error"].idxmin()].to_dict()
            max_error = event.loc[event["error"].idxmax()].to_dict()
  
            # calculate the real error @ 100 % for the selected strategy, which is the real error
            error_of_min_at100 = pd.DataFrame(error_at_100[(error_at_100["quantifier"] == min_error["quantifier"]) 
                                            & (error_at_100["C"] == min_error["C"])
                                            & (error_at_100["class_weight"] == min_error["class_weight"])])
            
            error_of_max_at100 = pd.DataFrame(error_at_100[(error_at_100["quantifier"] == max_error["quantifier"]) 
                                                        & (error_at_100["C"] == max_error["C"])
                                                        & (error_at_100["class_weight"] == max_error["class_weight"])])

            # TODO compute how many evaluations have been accepted; this is the cost
            # of the early stopping strategy
            n_evals_of_min = strategy_data[(strategy_data["quantifier"] == min_error["quantifier"])
                                        & (strategy_data["C"] == min_error["C"])
                                        & (strategy_data["class_weight"] == min_error["class_weight"])]["accepted"].sum()
            
            n_evals_of_max = strategy_data[(strategy_data["quantifier"] == max_error["quantifier"])
                                                    & (strategy_data["C"] == max_error["C"])
                                                    & (strategy_data["class_weight"] == max_error["class_weight"])]["accepted"].sum()
            
            # TODO store the results (error and number of evaluations)
            best_performance_one_run.append({
                "strategy": strategy_name,
                "fold_nr": i,
                "error@100": float(error_of_min_at100.iloc[0]["error"]),
                "quantifier@100": error_of_min_at100.iloc[0]["quantifier"],
                "C@100": error_of_min_at100.iloc[0]["C"],
                "class_weight@100": error_of_min_at100.iloc[0]["class_weight"],
                "n_evaluations": n_evals_of_min
            })

            worst_performance_one_run.append({
                            "strategy": strategy_name,
                            "fold_nr": i,
                            "error@100": float(error_of_max_at100.iloc[0]["error"]),
                            "quantifier@100": error_of_max_at100.iloc[0]["quantifier"],
                            "C@100": error_of_max_at100.iloc[0]["C"],
                            "class_weight@100": error_of_max_at100.iloc[0]["class_weight"],
                            "n_evaluations": n_evals_of_max
                        })
            
        return best_performance_one_run, worst_performance_one_run

    # experiment with all strategies
    parallel = Parallel(n_jobs=n_jobs, prefer="processes") # time for a test run: ~220s
    for run in parallel(delayed(eval_strategy)(data=data, error=error, folds=folds, quantifier=quantifier, val_samples=val_samples, batch_size=batch_size, strategy_name=strategy_name, strategy=strategy) for (strategy_name, strategy) in stopping_strategies.items()):
            best_performance.extend(run[0])
            worst_performance.extend(run[1])

                    
    best_performance = pd.DataFrame(best_performance)
    worst_performance = pd.DataFrame(worst_performance)

    averaged_best = best_performance.groupby("strategy")[["error@100", "n_evaluations"]].mean().reset_index()
    averaged_worst = worst_performance.groupby("strategy")[["error@100", "n_evaluations"]].mean().reset_index()

    maxs_b = (best_performance.groupby("strategy", as_index=False)["n_evaluations"].max().rename(columns={"n_evaluations": "max_n"}))
    mins_b = (best_performance.groupby("strategy", as_index=False)["n_evaluations"].min().rename(columns={"n_evaluations": "min_n"}))
    counts_b = (best_performance.groupby("strategy").apply(
        lambda df: (df[["quantifier@100", "C@100", "class_weight@100"]].apply(tuple, axis=1).value_counts().to_dict())
    , include_groups=False).rename("config_counts").reset_index())

    averaged_best = averaged_best.merge(maxs_b, on="strategy")
    averaged_best = averaged_best.merge(mins_b, on="strategy")
    averaged_best = averaged_best.merge(counts_b, on="strategy")

    maxs_w = (worst_performance.groupby("strategy", as_index=False)["n_evaluations"].max().rename(columns={"n_evaluations": "max_n"}))
    mins_w = (worst_performance.groupby("strategy", as_index=False)["n_evaluations"].min().rename(columns={"n_evaluations": "min_n"}))
    counts_w = (worst_performance.groupby("strategy").apply(
        lambda df: (df[["quantifier@100", "C@100", "class_weight@100"]].apply(tuple, axis=1).value_counts().to_dict())
        , include_groups=False).rename("config_counts").reset_index())
    
    averaged_worst = averaged_worst.merge(maxs_w, on="strategy")
    averaged_worst = averaged_worst.merge(mins_w, on="strategy")
    averaged_worst = averaged_worst.merge(counts_w, on="strategy")
    
    averaged_best.to_csv(quantifier + "_best" + ".csv")
    averaged_best.to_csv(quantifier + "_worst" + ".csv")

def plot(agg_res_b, agg_res_w):
    ######################################################
    ### Plotting ###
    ######################################################

    # plot the diagrams for the random stopping
    random_strategies = ["10%random", "20%random", "30%random", "40%random", "50%random", "60%random", "70%random", "80%random", "90%random", "100%random"]
    x = np.linspace(0.1, 1, 10)
    mean_error = []
    min_error = []
    max_error = []

    # for strat in random_strategies:
    #     mean = agg_res.loc[agg_res["strategy"] == strat, "error@100"].iloc[0]
    #     mean_error.append(mean)
    #     keys = list(agg_res.loc[agg_res["strategy"] == strat, "config_counts"].iloc[0].keys())
    #     items = list(agg_res.loc[agg_res["strategy"] == strat, "config_counts"].iloc[0].items())
    #     mask = error_at_100.apply(lambda row: (row["quantifier"], row["C"], row["class_weight"]) in keys, axis=1)
    #     min_error.append(abs(error_at_100.loc[mask, "error"].min()-mean))
    #     max_error.append(abs(error_at_100.loc[mask, "error"].max()-mean))

    # plt.figure(figsize=(12, 8))
    # plt.grid(alpha=0.4)
    # plt.tight_layout(pad=4.0)
    # plt.errorbar(x, mean_error, yerr=[min_error, max_error], fmt="o", color="red", ecolor="black", capsize=4)
    # plt.axhline(y = random_min_val, color="gray", linestyle="--")
    # plt.xlabel("Percentage of Data")
    # plt.ylabel(error + " (@100%)")
    # plt.title("Additional Error\n of selecting other than the best configurations")
    # plt.savefig("plots_percentage_" + file_time + ".png")
    # plt.close()

    def scatter(df_b, df_w):
        plt.figure(figsize=(12, 8))
        plt.grid(alpha=0.4)
        plt.tight_layout(pad=4.0)
        plt.scatter(df_b["n_evaluations"], df_b["error@100"], s=30,c="tab:red",marker="o", alpha=0.7)
        plt.scatter(df_w["n_evaluations"], df_w["error@100"], s=30,c="tab:red",marker="o", alpha=0.7)
        plt.xlabel("Number of evaluations")
        plt.ylabel(error + " (@100%)")
        plt.title(f"Best and Worst performance of Random Stop")
        plt.savefig("best_worst_random_" + error + ".png")
        plt.close()
    
    # one scatter plot for best/worst random
    random_b = agg_res_b[agg_res_b["strategy"].str.endswith("random")]
    random_w = agg_res_w[agg_res_w["strategy"].str.endswith("random")]
    scatter(random_b, random_w)



# use file baseline_2026_7_17_9_49_49_lequa2022_T1B.csv as a test
# file baseline_2026_7_1_15_40_17_lequa2022_T1B.csv is whole set
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='min_max_random.py',
                    description='Run stopping Experiment and save data',
                    epilog='see other resources')
    parser.add_argument("filename", help="path to the directory of the saved data", type=str)
    parser.add_argument("-e", "--error_metric", help="error metric to be looked at", type=str)
    parser.add_argument("-f", "--folds", help="number of folds for calculating mean and variance", type=int)
    parser.add_argument("-j", "--jobs", help="number of parallel jobs", type=int)
    parser.add_argument("--test", help="toggle test mode", action="store_true")
    

    args = parser.parse_args()
    print("Running script:" + parser.prog)
    file = args.filename
    error = args.error_metric
    folds = args.folds
    test_flag = args.test
    n_jobs = args.jobs
    
    if test_flag:
        quantifier = ["ACC"]
    else:
        quantifier = ["ACC", "PACC", "SLD"]

    for q in quantifier:
        data = unpack(file, q)
        experiment(data, error, folds, q, n_jobs, batch_size_factor=0.01, test_flag=test_flag)

    res_acc_b = unpack_result("ACC_best.csv")
    res_pacc_b = unpack_result("PACC_best.csv")
    res_sld_b = unpack_result("SLD_best.csv")
    results_b = pd.concat([res_acc_b, res_pacc_b, res_sld_b], ignore_index=True)
    
    stacked_b = pd.concat([res_acc_b, res_pacc_b, res_sld_b], keys=range(3))
    result_b = (stacked_b.groupby(level=1).apply(lambda g: g.loc[g["error@100"].idxmin()]).reset_index(drop=True))
    result_b.to_csv(f"all_best_best_{error}.csv")

    res_acc_w = unpack_result("ACC_worst.csv")
    res_pacc_w = unpack_result("PACC_worst.csv")
    res_sld_w = unpack_result("SLD_worst.csv")
    results_w = pd.concat([res_acc_w, res_pacc_w, res_sld_w], ignore_index=True)
    
    stacked_w = pd.concat([res_acc_w, res_pacc_w, res_sld_w], keys=range(3))
    result_w = (stacked_w.groupby(level=1).apply(lambda g: g.loc[g["error@100"].idxmax()]).reset_index(drop=True))
    result_w.to_csv(f"all_worst_worst_{error}.csv")

    plot(result_b, result_w)
