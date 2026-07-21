import argparse
import time
from collections import defaultdict
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quapy as qp
from stopping_instanceSelection import RandomStop
from stopping_instanceSelection import RankingStop
from stopping_instanceSelection import EBGstop
from stopping_instanceSelection import WilcoxonStop
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

# Main method for running the experiment
def experiment(data, error, folds, quantifier, batch_size_factor = 0.01, test_flag=False):
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
        batch_size = 10
    print(f"The dataset has {len(val_samples)} many validation samples and a batchsize of {batch_size}")

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

    for i in [3, 6, 0]:
        for l in [2, 8, 10, 12]: # number of starting samples
            strategy_name = f"top{i}ranking_5"
            stopping_strategies[strategy_name] = RankingStop(
                ["quantifier", "C", "class_weight"],
                num_iterations=5, # 5*batchsize
                error=error,
                number_equal_configs=i # just count the rank of the best i configs (0 = all configs)
            )
            strategy_name = f"top{i}ranking_10"
            stopping_strategies[strategy_name] = RankingStop(
                ["quantifier", "C", "class_weight"],
                num_iterations=10, # 10*batchsize
                error=error,
                number_equal_configs=i # just count the rank of the best i configs (0 = all configs)
            )
            strategy_name = f"top{i}ranking_20"
            stopping_strategies[strategy_name] = RankingStop(
                ["quantifier", "C", "class_weight"],
                num_iterations=20, # 20*batchsize
                error=error,
                n_samples=len(val_samples),
                min_samples=l,
                number_equal_configs=i # just count the rank of the best i configs (0 = all configs)
            )
    
    for j in [0.01, 0.05, 0.1]:
        for m in [2, 8, 10, 12]:
            strategy_name = f"wilcoxon{j}_{m}"
            stopping_strategies[strategy_name] = WilcoxonStop(
                ["quantifier", "C", "class_weight"],
                error=error,
                n_samples=len(val_samples),
                min_samples=m,
                p_threshold=j
            )
    
    for k in [0.01, 0.05, 0.1]:
        for n in [2, 8, 10, 12]:
            strategy_name = f"EBGstop{k}_n"
            stopping_strategies[strategy_name] = EBGstop(
                ["quantifier", "C", "class_weight"],
                error=error,
                n_samples=len(val_samples),
                min_samples=n,
                delta = 0.1,
                epsilon = k
            )

    # TODO other strategies should be added as soon as they are implemented

    # Calculate the performance value and index for each configuration on each fold of each fraction of data
    best_performance = []
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

    # experiment with all strategies
    for strategy_name, strategy in stopping_strategies.items():
        print(f"This is stopping strategy {strategy_name}")
        rng = np.random.default_rng(42)
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
  
            # calculate the real error @ 100 % for the selected strategy, which is the real error
            error_of_min_at100 = pd.DataFrame(error_at_100[(error_at_100["quantifier"] == min_error["quantifier"]) 
                                            & (error_at_100["C"] == min_error["C"])
                                            & (error_at_100["class_weight"] == min_error["class_weight"])])

            # TODO compute how many evaluations have been accepted; this is the cost
            # of the early stopping strategy
            n_evals_of_min = strategy_data[(strategy_data["quantifier"] == min_error["quantifier"])
                                        & (strategy_data["C"] == min_error["C"])
                                        & (strategy_data["class_weight"] == min_error["class_weight"])]["accepted"].sum()
            
            # TODO store the results (error and number of evaluations)
            best_performance.append({
                "strategy": strategy_name,
                "fold_nr": i,
                "error@100": float(error_of_min_at100.iloc[0]["error"]),
                "quantifier@100": error_of_min_at100.iloc[0]["quantifier"],
                "C@100": error_of_min_at100.iloc[0]["C"],
                "class_weight@100": error_of_min_at100.iloc[0]["class_weight"],
                "n_evaluations": n_evals_of_min
            })
        
    best_performance = pd.DataFrame(best_performance)

    averaged_best = best_performance.groupby("strategy")[["error@100", "n_evaluations"]].mean().reset_index()

    maxs = (best_performance.groupby("strategy", as_index=False)["n_evaluations"].max().rename(columns={"n_evaluations": "max_n"}))
    mins = (best_performance.groupby("strategy", as_index=False)["n_evaluations"].min().rename(columns={"n_evaluations": "min_n"}))
    counts = (best_performance.groupby("strategy").apply(
        lambda df: (df[["quantifier@100", "C@100", "class_weight@100"]].apply(tuple, axis=1).value_counts().to_dict())
    , include_groups=False).rename("config_counts").reset_index())

    averaged_best = averaged_best.merge(maxs, on="strategy")
    averaged_best = averaged_best.merge(mins, on="strategy")
    averaged_best = averaged_best.merge(counts, on="strategy")
        
    # for now, we do not need a plot, just a table
    print(best_performance)
    print(averaged_best)

    best_performance.to_csv(filename + "lines" + ".csv")
    averaged_best.to_csv(filename + "agg" + ".csv")

    return best_performance


# use file baseline_2026_7_17_9_49_49_lequa2022_T1B.csv as a test
# file baseline_2026_7_1_15_40_17_lequa2022_T1B.csv is whole set
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='stopping_experiment.py',
                    description='Run stopping Experiment and save data',
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
    test_flag = args.test
    
    if test_flag:
        quantifier = ["ACC"]
    else:
        quantifier = ["ACC", "PACC", "SLD"]

    for q in quantifier:
        data = unpack(file, q)
        experiment(data, error, folds, q, batch_size_factor=0.01, test_flag=test_flag)
