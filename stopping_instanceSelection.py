import jax
import jax.numpy as jnp
import pandas as pd
import argparse
import time
import scipy as sc
from scipy.stats import wilcoxon
from abc import ABC, abstractmethod

MAX_INT = jnp.int64(jnp.iinfo(jnp.int64).max)

class Stopping(ABC):
    @abstractmethod
    def decide(self):
        pass

    @abstractmethod
    def update(self):
        pass


class EBGstop(Stopping):
    def __init__(self, attr):
        self.attr = attr
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

    def algo(data_array, i,beta = 1.1, eps = 0.1, delta = 0.1, p = 1.1):
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

class WilcoxonStop(Stopping):
    def __init__(self, attr):
        self.attr = attr
        
    def algo():
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

class RankingStop(Stopping):
    def __init__(self, attr):
        self.attr = attr

    def algo():
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
        
class TStop(Stopping):
    def __init__(self, attr):
        self.attr = attr

    def algo():
        pass




# Methods for instance selection

class InstanceSelection(ABC):
    @classmethod
    def sampling():
        pass
    # Instance selection methods

class BaselineSampling(InstanceSelection):
    def sampling():
        pass

class DiscriminationSampling(InstanceSelection):
    def sampling():
        pass

class VarianceSampling(InstanceSelection):
    def sampling():
        pass

class UDDSampling(InstanceSelection):
    def sampling():
        pass

class UncertaintySampling(InstanceSelection):
    def sampling():
        pass

class InformationSampling(InstanceSelection):
    def sampling():
        pass

class FeatureSampling(InstanceSelection):
    def sampling():
        pass

