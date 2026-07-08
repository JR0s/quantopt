import argparse
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import quapy as qp
from stopping_instanceSelection import RandomStop
from stopping_instanceSelection import BaselineSampling

# Method to load the specified file for a given quantifier
def unpack(file, quantifier):
    df = pd.read_csv(file, index_col=0)
    data = df.loc[df["quantifier"] == quantifier]
    data["p_est"] = data["p_est"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    data["p_val"] = data["p_val"].apply(lambda x: np.array(x.strip("[]").split(), dtype=float))
    # class_weight is saved as na but the hyperparameter has value None (string here, type None leads to errors)
    data["class_weight"] = data["class_weight"].where(data["class_weight"].notna(), "None")
    return data

# Method for plotting the additional error of selecting different configurations at lower percentages than the best performing configuration with 100% of data
def plot(data, error, folds, quantifier):
    file_time = time.localtime()
    file_time = str(file_time.tm_year) + "_" + str(file_time.tm_mon) + "_" + str(file_time.tm_mday) + "_" + str(file_time.tm_hour) + "_" + str(file_time.tm_min) + "_" + str(file_time.tm_sec)
    filename = "percentage_sampling_" + quantifier + "_" + file_time + "_" + ".csv"

    # how many configurations and which validation samples are there?
    n_configurations = len(data[["C", "class_weight"]].drop_duplicates())
    val_samples = pd.unique(data["val_sample"])

    # how many samples are acquired in each round?
    batch_size = 10 # could be another number

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

    # instantiate all strategies for early stopping
    stopping_strategies = {} # map from strategy name (key) to a Stopping object (value)
    percentages = np.linspace(0.1, 1, 10)
    for percentage in percentages:
        strategy_name = f"{int(percentage*100)}%random"
        stopping_strategies[strategy_name] = RandomStop(
            ["quantifier", "C", "class_weight"],
            percentage * len(val_samples),
        )
    
    # TODO other strategies should be added as soon as they are implemented

    sampling_strategies = {}
    sampling_name = "baseline"
    sampling_strategies[sampling_name] = BaselineSampling(data, batch_size, batch_size)
    # Calculate the performance value and index for each configuration on each fold of each fraction of data
    best_performance = []
    # we want to look at folds per stopping per configuration

    # randomization is not needed yet (only when adding folds)
    rand = np.random.RandomState(seed=42)

    # experiment with all strategies
    for strategy_name, strategy in stopping_strategies.items():
        print(f"This is stopping strategy {strategy_name}")
        # copy the data so that it can be split without breaking the original object
        strategy_data = data.copy()

        # keep track of which evaluations are accepted by the strategy
        strategy_data["accepted"] = False

        # accept the first N evaluations to initialize the strategy
        initial_samples = val_samples[:batch_size]
        strategy_data.loc[strategy_data["val_sample"].isin(initial_samples), "accepted"] = True
        strategy(strategy_data[strategy_data["accepted"]]) # first data for strategy
        sampler = BaselineSampling(strategy_data, batch_size, batch_size)

        strategy_data["stopped"] = False

        while((not all(strategy_data["stopped"] == True)) and sampler.iter < sampler.length):
            iteration_samples = sampler.sampling()
            stopped = strategy(strategy_data[strategy_data["accepted"]])
            #print(f"length of stopped: {len(stopped)}") # = 120
            #print(f"length of strategy_data:{len(strategy_data)}")# = 1212
            strategy_data = pd.merge(left=strategy_data.drop(columns=["stopped"]), right=stopped.drop(columns=["p_est", "p_val", "t_est", "t_train", "accepted"]), on=("quantifier", "C", "class_weight", "val_sample"), how="left")
            #strategy_data["stopped"] = stopped["stopped"] # length mismatch -> look at lengths
            # merge over configs + join on right side and left side with rest without stop
            strategy_data.loc[strategy_data["val_sample"].isin(iteration_samples) & (strategy_data["stopped"] == False), "accepted"] = True

        strategy_data.to_csv(f"join_test_{strategy_name}"+ file_time)
        print(strategy_data)
        # the strategy has now stopped all configurations, so that we can evaluate

        # among all accepted evaluations, compute the apparent error
        event = strategy_data[strategy_data["accepted"]].groupby(["C", "class_weight"]).apply(compute_error, include_groups=False)
        event = pd.DataFrame(event, columns=["error"]).reset_index()
        
        # TODO find the best configuration according to the apparent error
        min_error = event.loc[event["error"].idxmin()].to_dict()        

        # TODO find the corresponding real error (@ 100 %) of this configuration; this
        # is the error of the early stopping strategy
        error_of_min_at100 = error_at_100[(error_at_100["C"] == min_error["C"]) & (error_at_100["class_weight"] == min_error["class_weight"])]

        # TODO compute how many evaluations have been accepted; this is the cost
        # of the early stopping strategy

        # TODO store the results (error and number of evaluations)
        best_performance.append({
            "strategy": strategy_name,
            "error": error_of_min_at100["error"],
            "n_evaluations": None,
        })
        
    best_performance = pd.DataFrame(best_performance)

    # for now, we do not need a plot, just a table
    print(best_performance)

    return best_performance


# use file baseline_2026_7_2_14_45_53_lequa2022_T1B.csv as a test
# file baseline_2026_7_1_15_40_17_lequa2022_T1B.csv is whole set
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
    test_flag = args.test
    
    if test_flag:
        quantifier = ["ACC"]
    else:
        quantifier = ["ACC", "PACC", "SLD"]

    for q in quantifier:
        data = unpack(file, q)
        plot(data, error, folds, q)

