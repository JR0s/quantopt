import jax
import jax.numpy as jnp
import pandas as pd
import argparse
import time
import scipy as sc
from scipy.stats import wilcoxon

MAX_INT = jnp.int64(jnp.iinfo(jnp.int64).max)

# method for unpacking the csv file
def unpacking(input_file):
    df = pd.read_csv(input_file)
    # split the runs of each parameter and quantification method



# stopping methods
# i -> ith method being analyzed
# beta ->
# eps ->
# delta ->
# p ->
# t ->
# k ->
# X ->
# alpha ->
# xs ->
# c ->
# lb -> lower bound
# ub -> upper_bound
# dk ->

def EBGstop(data_array, i,beta = 1.1, eps = 0.1, delta = 0.1, p = 1.1):
    c = 0
    lb = 0
    ub = MAX_INT
    t = 1
    k = 0
    X = data_array[i][0]
    while((1+eps)*lb < (1-eps)*ub):
        t = t+1
        X = data_array[i][t]
        if(t > jnp.floor(beta**k)):
            k = k+1
            alpha = jnp.floor(beta**k)/jnp.floor(beta**(k-1))
            dk = c/jnp.log(beta, t)**p
            xs = -alpha*jnp.log(dk/3)
        sigma = 42 # ändern in die richtige berechnung von sigma
        R = 42 # ändern in die richtige berechnung von R
        c = sigma*jnp.sqrt(2*xs/t) + 3*R*xs/t
        lb = max(lb, jnp.abs(X) - c)
        ub = min(ub, jnp.abs(X) + c)
    
    return jnp.sign(X)*0.5*((1+eps)*lb + (1-eps)*ub)



#
# wilcoxon_history ->
# y_sampled ->
# threshold ->
# ema ->
# ema_val ->
# v ->
# p_val_thresh ->

def WilcoxonStop():
    wilcoxon_history = []
    y_sampled = []
    threshold = 0
    p_val_thresh = 0
    x = []
    y = []
    ema = wilcoxon(x,y)
    if(len(wilcoxon_history) == 0 or jnp.count_nonzero(y_sampled)/y_sampled.shape[0] < threshold):
        return False
    else:
        ema_val = wilcoxon_history[0]
        for v in wilcoxon_history[1:]:
            ema_val = ema * v + (1 - ema) * ema_val
        return ema_val <= p_val_thresh


#
# ranking_history ->
# con_num_instances ->
# min_amount ->
# y_sampled ->
#
#

def RankingStop():
    ranking_history = 0
    con_num_instances = 0
    min_amount = 0
    y_sampled = []
    if (len(ranking_history) < con_num_instances or
            jnp.count_nonzero(y_sampled) / y_sampled.shape[0] < min_amount):
        return False
    else:
        recent = ranking_history[-1]
        for i in range(2, con_num_instances + 1):
            if not jnp.all(recent == ranking_history[-i]):
                return False
        return True


# Instance selection methods

def BaselineSampling():
    pass


def DiscriminationSampling():
    pass


def VarianceSampling():
    pass


def UDDSampling():
    pass


def UncertaintySampling():
    pass


def InformationSampling():
    pass


def FeatureSampling():
    pass



# coordinating the selection of stopping methods and plotting results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='stopping_instanceSelection.py',
                    description='This is the part of the thesis where the stopping and instance selection takes place')
    parser.add_argument("input_file", help="dataset which is fetched from quapy", type=str)
    
    args = parser.parse_args()
    print("Running script:" + parser.prog)
    file_in = args.input_file

    data = unpacking(file_in)


