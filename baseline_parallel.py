import argparse
import time
import itertools
from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import quapy as qp
from quapy.method.aggregative import ACC, PACC, SLD
from sklearn.linear_model import LogisticRegression
from quapy.data.datasets import LEQUA2022_SAMPLE_SIZE, LEQUA2024_SAMPLE_SIZE

# Method that initialises the dataset
def prepare_dataset(dataset):
    # load the dataset
    if(type(dataset) == tuple):
        name, task = dataset
        fetch = getattr(qp.datasets, "fetch_" + name)
        training, val_generator, test_generator = fetch(task=task)
        match name:
            case "lequa2022" : qp.environ["SAMPLE_SIZE"] = LEQUA2022_SAMPLE_SIZE[task]
            case "lequa2024" : qp.environ["SAMPLE_SIZE"] = LEQUA2024_SAMPLE_SIZE[task]
    else:
        name = dataset
        task = None
        fetch = getattr(qp.datasets, "fetch_" + name)
        training, val_generator, test_generator = fetch()
    
    # split the set into training, validation and test subsets
    return training, val_generator, test_generator

# Method that defines a single run for one configuration
def generation_run(train,val, test, test_flag, i, quantifier_params):
    one_res = []
    model_C, class_we = quantifier_params
    # define the quantifier for each run
    match i:
        case 0: 
            model = ACC(LogisticRegression(C=model_C, class_weight=class_we, max_iter=10 if test_flag else 1000, n_jobs=1))
            model_name = "ACC"
        case 1:
            model = PACC(LogisticRegression(C=model_C, class_weight=class_we, max_iter=10 if test_flag else 1000,n_jobs=1))
            model_name="PACC"
        case 2:
            model = SLD(LogisticRegression(C=model_C, class_weight=class_we, max_iter=10 if test_flag else 1000,n_jobs=1))
            model_name="SLD"
        case _: raise ValueError("Error while iterating quantifiers.")
            
    print("Training the quantifier: " + str(model))
        
    # train the quantifier
    t_train_begin = time.time()
    xtr, ytr = train.Xy
    trained_quantifier = model.fit(xtr, ytr)
    t_train = time.time() - t_train_begin
    print("Training took: " + str(t_train))

    # evaluate the quantifier and report the results
    for test_run_number, (X_i, p_i) in enumerate(val()):
        t0 = time.time()
        p_est = trained_quantifier.predict(X_i)
        t_est = time.time() - t0
        one_res.append({
            "quantifier": model_name,
            "C": model_C,
            "class_weight": class_we,
            "p_est": p_est,
            "p_val": p_i,
            "t_est": t_est,
            "t_train": t_train,
            "val_sample": test_run_number
        })
        if test_flag and test_run_number >= 100: # select the number of iterations for a test run
            break
    return one_res
    


# Function for a given experiment
def baseline_experiment(dataset, n_jobs, test_flag=False):
    if(type(dataset) == tuple):
        name, task = dataset
        name = name + "_" + task
    else:
        name = dataset
    file_time = time.localtime()
    file_time = str(file_time.tm_year) + "_" + str(file_time.tm_mon) + "_" + str(file_time.tm_mday) + "_" + str(file_time.tm_hour) + "_" + str(file_time.tm_min) + "_" + str(file_time.tm_sec)
    filename = "baseline_" + file_time + "_" + name + ".csv"

    # load the dataset
    train, val, test = prepare_dataset(dataset)
    
    # Define the quantifier/classifier parameters
    # parameters of the type: C, class_weight
    model_C = np.geomspace(1e-3, 1e2, 6 if test_flag else 21)
    class_w = [None, "balanced"]
    quantifier_params = list(itertools.product(model_C, class_w))

    results = []
    parallel = Parallel(n_jobs=n_jobs, prefer="processes") # time for a test run: ~220s
    #parallel = Parallel(n_jobs=n_jobs, prefer="threads") # time for a test run: ~456s
    quant_number = [0] if test_flag else[0,1,2]
    for i in quant_number: # tune for the nuber of quantifiers
        for run in parallel(delayed(generation_run)(train, val, test,test_flag, i, params) for params in quantifier_params):
            results.extend(run)

    results = pd.DataFrame(results)
    results.to_csv(filename)
    return results
    

# main method for the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='baseline_experiment.py',
                    description='This is the baseline experiment of the thesis',
                    epilog='see other resources')
    parser.add_argument("dataset_name", help="dataset which is fetched from quapy", type=str)
    parser.add_argument("-t", "--task", help="name of the task of a given dataset specified in dataset_name", type=str)
    parser.add_argument("-j", "--n_jobs", help="pass the number of available cores", type=int)
    parser.add_argument("--test", help="toggle test mode", action="store_true")

    args = parser.parse_args()
    print("Running script:" + parser.prog)
    dataset = args.dataset_name
    if args.task is not None:
        dataset = (args.dataset_name, args.task)

    t0 = time.time()
    baseline_experiment(dataset, args.n_jobs, args.test)
    t_end = time.time()-t0
    print(f"experiment took {t_end} seconds")
