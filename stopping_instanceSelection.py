import numpy as np
import pandas as pd
import math
from quapy.error import mae, mrae, mkld

from scipy.stats import wilcoxon
from abc import ABC, abstractmethod


class Stopping(ABC):
    @abstractmethod
    def __call__(self, sampled_value): # returns stopping decision
        pass


class RandomStop(Stopping):
    def __init__(self, config_columns, n_samples):
        self.config_columns = config_columns
        self.n_samples = n_samples

    def __call__(self, dataframe):
        data = dataframe.copy()
        data["stopped"] = False
        data["stopped"] = data.groupby(self.config_columns)["val_sample"].transform(lambda gdf: gdf.nunique() >=  self.n_samples)
        return data

    

class EBGstop(Stopping):
    def __init__(self, config_columns, error, n_samples, min_samples = 0.02, delta = 0.01, epsilon = 0.01):
        self.beta = 1.1 # factor of taking too many samples in geometric sampling
        self.delta = delta # allowed error on the estimated mean
        self.epsilon = epsilon # corresponding epsilon to the delta for epsilon-delta criterium
        self.p = 1.1 # scaling factor for d
        self.range = {} # range of the random value
        self.t = {} # t-th taken sample
        self.k = {} # updating factor for geometric sampling
        self.alpha = {} # factor in geometric sampling: fraction of samples from two following iterations
        self.x = {} # variable in calculation
        self.c = self.delta*(self.p-1)/self.p # constant factor for d_t
        self.error = error
        self.config_columns = config_columns
        self.init = True
        self.pred_mean = {}
        self.min_samples = min_samples
        self.n_samples = n_samples

        # bounds
        self.lb = {}
        self.ub = {}

        # values for tracking and updating the mean and variance
        self.error_values = {}
        self.x_mean_array = {} # predicted mean over i samples
        self.variance_array = {} # variance of the mean over i samples
        
        # values for the Welford's algorithm
        self.mean = {}
        self.M2 = {}
        self.mean_sum = {}
        self.var_sum = {}

        # track the values for the stopping decision
        self.ct_history = {}
        self.samples = {}
        
    # use Welford's algorithm to udpdate the mean and variance
    def update(self, config, val_sample, sample_err):
        self.error_values[config].append(sample_err)
        self.t[config] = self.t[config] + 1

        old_mean = self.mean[config]
        self.mean[config] = self.mean[config] + (sample_err - self.mean[config])/self.t[config]
        self.M2[config] = self.M2[config] + (sample_err - old_mean)*(sample_err - self.mean[config])

        self.x_mean_array[config].append(self.mean[config])
        self.variance_array[config].append(self.M2[config]/self.t[config])
        self.mean_sum[config] = 1/self.t[config]*sum(self.x_mean_array[config])
        summe = 0
        for i in self.x_mean_array[config]:
            summe += (i - self.mean_sum[config])**2
        self.var_sum[config] = np.sqrt(summe/self.t[config])
        # print(f"config = {config}, val_sample = {val_sample}, err_val = {sample_err}, t = {self.t[config]}, mean = {self.mean[config]}, M2 = {self.M2[config]}, mean in array = {self.x_mean_array[config][-1]}, var = {self.variance_array[config][-1]}, 1/t sum of mean = {self.mean_sum[config]}, 1/t var = {self.var_sum[config]}")
    
    def compute_error(self, p_est, p_val):
        match self.error:
            case "mae":
                return mae(p_est, p_val)
            case "mrae":
                return mrae(p_est, p_val, eps=1/(2*len(p_est)))
            case "mkld":
                return mkld(p_est, p_val, eps=1/(2*len(p_est)))

    # abhängig von config: t, k, x, alpha
    # nicht abhängig: beta, delta, epsilon, p, range, c

    def algo(self, config, val_sample, sample_err):
        # calculate the ith predicted mean and its variance
        self.update(config, val_sample, sample_err)
        dk = 0
        if(self.t[config] >= np.floor(self.beta**self.k[config])):
            self.k[config] = self.k[config]+1
            self.alpha[config] = np.floor(self.beta**self.k[config])/np.floor(self.beta**(self.k[config]-1))
            #dk = self.c/(self.k[config]**self.p) # used in EB stop -> d = series of probabilities going to delta
            dk = self.c / (math.log(self.beta, self.t[config])**self.p) # used in ebgstop
            self.x[config] = -self.alpha[config]*np.log(dk/3)
        self.range[config] = self.ub[config] - self.lb[config] # update range of intervall
        ct = self.var_sum[config]*np.sqrt(2*self.x[config]/self.t[config]) + 3*self.range[config]*self.x[config]/self.t[config]
        self.ct_history[config].append(ct)
        
        self.lb[config] = max(self.lb[config], abs(self.mean_sum[config]) - self.ct_history[config][-1])
        self.ub[config] = min(self.ub[config], abs(self.mean_sum[config]) + self.ct_history[config][-1])
        # print(f"t = {self.t[config]}, k = {self.k[config]}, x = {self.x[config]}, alpha = {self.alpha[config]}, dk = {dk}, ct = {ct}, lb = {self.lb[config]}, ub = {self.ub[config]}")
        
        self.pred_mean[config] = 0.5 * ((1 + self.epsilon) * self.lb[config] + (1 - self.epsilon) * self.ub[config])
        return self.lb[config] * (1 + self.epsilon) >= self.ub[config] * (1 - self.epsilon) # continue as long as the lower bound is smaller
        
    def __call__(self, dataframe):
        data = dataframe.copy()

        # method needs to have a certain minimum amount of samples before stopping
        current_n_samples = data["val_sample"].nunique()
        if(current_n_samples <= int(self.min_samples*self.n_samples)):
            data["stopped"] = False
            return data
        
        # consider only not stopped configurations
        mask = (data.groupby(self.config_columns)["stopped"].transform("any"))
        data = data[~mask]
        # print(data.groupby(self.config_columns).ngroups)
        if(data.groupby(self.config_columns).ngroups == 1):
            data["stopped"] = True
            return data
        
        # apply the algorithm to each configuration individually
        for config, group in data.groupby(self.config_columns):
            if(self.init): # first time: fill dict with config values and initialize the samples -> do look at each sample just once
                self.samples[config] = []
                self.t[config] = 0
                self.k[config] = 0
                self.x[config] = 0
                self.alpha[config] = 0
                self.lb[config] = 0
                self.ub[config] = 10000000
                self.ct_history[config] = []
                self.error_values[config] = []
                self.range[config] = 1

                self.x_mean_array[config] = []
                self.variance_array[config] = []
                self.mean[config] = 0
                self.M2[config] = 0
                self.mean_sum[config] = 0
                self.var_sum[config] = 0
                self.pred_mean[config] = 0
            
            stopped = False # initialise the stopping variable
            for row in group.itertuples(index=False): # iterate over each sample
                if(not (row.val_sample in self.samples[config])):
                    self.samples[config].append(row.val_sample)
                    err = self.compute_error(row.p_est, row.p_val)
                    if(len(self.error_values[config]) == 0):
                        self.update(config, row.val_sample, err)
                    else:
                        stopped = self.algo(config, row.val_sample, err)
                    if(stopped == True):
                        break
            mask = (
                    (data["quantifier"] == config[0]) &
                    (data["C"] == config[1]) &
                    (data["class_weight"] == config[2])
                )
            data.loc[mask, "stopped"] = stopped
        self.init = False # set self.init to False after the first call
        return data

# literature: minimum samples: [2, 8, 10, 12]% of samples before applying criterion
# p-value dropoff: 5%
# exponential moving average with beta = [0.1, 0.7]
class WilcoxonStop(Stopping):
    def __init__(self, config_columns, error, n_samples, min_samples=0.02, p_threshold=0.05):
        self.error = error
        self.config_columns = config_columns
        self.p_threshold = p_threshold
        self.best_config = None
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.configs = {}

    def compute_error(self, p_est, p_val):
        match self.error:
            case "mae":
                return mae(p_est, p_val)
            case "mrae":
                return mrae(p_est, p_val, eps=1/(2*len(p_est)))
            case "mkld":
                return mkld(p_est, p_val, eps=1/(2*len(p_est)))
        

    def __call__(self, dataframe):
        data = dataframe.copy()

        # method needs to have a certain minimum amount of samples before stopping
        current_n_samples = data["val_sample"].nunique()
        if(current_n_samples <= int(self.min_samples*self.n_samples)):
            data["stopped"] = False
            return data

        # consider only not stopped configurations
        mask = (data.groupby(self.config_columns)["stopped"].transform("any"))
        data = data[~mask]
        if(data.groupby(self.config_columns).ngroups == 1):
            data["stopped"] = True
            return data
        val_error = data.groupby(self.config_columns+["val_sample"]).apply(lambda g: self.compute_error(g["p_est"], g["p_val"]), include_groups=False).rename("val_error").reset_index()
        data = data.merge(val_error, on=self.config_columns+["val_sample"], how="left")
        errors = data.groupby(self.config_columns).apply(lambda g: list(zip(g["val_sample"], g["val_error"])), include_groups=False).to_dict()

        min_err = {}
        for key,values in errors.items():
            min_err[key] = np.mean([x[1] for x in values])

        self.best_config = (min(min_err.items(), key=lambda x: x[1]))
        self.best_config = (self.best_config[0], [x[1] for x in errors[self.best_config[0]]])
        errors.pop(self.best_config[0], None)

        for configuration, values in errors.items():
            y = [x[1] for x in values]
            difference = np.asarray(self.best_config[1]) - np.asarray(y)
            if np.allclose(difference, 0): # handle zero_method exception for wilcoxon test, if the values somehow are of same value
                p_value = 1.0
            else:
                w_statistic, p_value = wilcoxon(x=self.best_config[1], y=y, alternative="less", method="exact") # use one sided wilcoxon test, zero_method="pratt" can be used as alternative
            if(p_value < self.p_threshold):
                mask = (
                    (data["quantifier"] == configuration[0]) &
                    (data["C"] == configuration[1]) &
                    (data["class_weight"] == configuration[2])
                )
                data.loc[mask, "stopped"] = True
                
        data = data.drop(columns=["val_error"])

        return data


# literature: minimum samples: [2, 8, 10, 12]% of samples before applying criterion
# convergence: stay the same for [1,2]% of all samples
class RankingStop(Stopping):
    def __init__(self, config_columns, num_iterations, error, n_samples, min_samples=0.02, number_equal_configs=0):
        self.config_columns = config_columns
        self.ranking_history = []
        self.num_iterations = num_iterations # number of iterations the ranking should stay the same before stopping
        self.counter = 0
        self.error = error
        self.min_samples = min_samples
        self.n_samples = n_samples
        self.number_equal_configs = number_equal_configs # set the number of considered configurations for stopping

    def compute_error(self, p_est, p_val):
        match self.error:
            case "mae":
                return mae(p_est, p_val)
            case "mrae":
                return mrae(p_est, p_val, eps=1/(2*len(p_est)))
            case "mkld":
                return mkld(p_est, p_val, eps=1/(2*len(p_est)))

    def __call__(self, dataframe):
        if(len(dataframe) == 0):
            raise ValueError("The passed dataframe is empty.")
        data = dataframe.copy()

        # method needs to have a certain minimum amount of samples before stopping
        current_n_samples = data["val_sample"].nunique()
        if(current_n_samples <= int(self.min_samples*self.n_samples)):
            data["stopped"] = False
            return data
        
        # perform just on not stopped configs
        mask = (data.groupby(self.config_columns)["stopped"].transform("any"))
        data = data[~mask]
        if(data.groupby(self.config_columns).ngroups == 1):
            data["stopped"] = True
            return data

        val_error = data.groupby(self.config_columns).apply(lambda g: self.compute_error(g["p_est"], g["p_val"]), include_groups=False).rename("val_error").reset_index()
        data = data.merge(val_error, on=self.config_columns, how="left")
        errors = data.groupby(self.config_columns)["val_error"].first().sort_values() # use first since the val errors are the same for each configuration
        error_list = list(errors.items())
        config_rank = []
        for config in error_list:
            config_rank.append(config[0])
        if(len(self.ranking_history) == 0):
            self.ranking_history = config_rank
            if(self.number_equal_configs == 0): # the ranking of all configurations is considered for stopping
                self.number_equal_configs = len(self.ranking_history)
            data = data.drop(columns=["val_error"])
            data["stopped"] = False
            return data

        if(self.ranking_history[:self.number_equal_configs] == config_rank[:self.number_equal_configs]): # before: if(np.all(self.ranking_history[:self.number_equal_configs] == config_rank[:self.number_equal_configs]))
            self.counter = self.counter + 1
        else:
            self.ranking_history = config_rank
            self.counter = 0
        data = data.drop(columns=["val_error"])
        data["stopped"] = (self.counter >= self.num_iterations)
        return data



# Methods for instance selection

class InstanceSelection(ABC):
    @classmethod
    def sampling():
        pass
    # Instance selection methods

class BaselineSampling(InstanceSelection):
    def __init__(self, dataset, batch_size, starting_index=0):
        self.iter = starting_index
        self.batch_size = batch_size
        self.length = len(dataset)
        self.data = dataset
        self.history = []
    def sampling(self):
        if(self.iter+self.batch_size < self.length):
            res = self.data.iloc[self.iter:self.iter+self.batch_size]
            self.history.extend(list(np.arange(self.iter, self.iter+self.batch_size, step=1)))
            self.iter = self.iter+self.batch_size
            return list(res["val_sample"])
        elif(self.iter < self.length):
            res = self.data.iloc[self.iter:self.length-1]
            self.history.extend(list(np.arange(self.iter, self.iter+self.batch_size, step=1)))
            self.iter = self.length
            return list(res["val_sample"])
        else:
            raise ValueError(f"Index {self.iter} is outside of Dataframe bound {self.length}")
        

class BaseSampling(InstanceSelection):
    def __init__(self, sample_array, batch_size, rng, starting_index=0):
        self.iter = starting_index
        self.batch_size = batch_size
        self.length = len(sample_array)
        self.data = rng.permutation(sample_array)
    def sampling(self):
        if(self.iter+self.batch_size <= self.length):
            res = self.data[self.iter:self.iter+self.batch_size]
            self.iter = self.iter+self.batch_size
            return res
        elif(self.iter < self.length):
            res = self.data[self.iter:]
            self.iter = self.length
            return res
        else:
            raise ValueError(f"Index {self.iter} is outside of Dataframe bound {self.length}")

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

