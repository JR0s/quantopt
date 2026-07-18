import numpy as np
import pandas as pd
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

    def get_attributes(self):
        return None

class EBGstop(Stopping):
    def __init__(self, beta=1.1, delta=0.1, epsilon = 0.1, p = 1.1, range = 1):
        self.beta = beta # factor of taking too many samples in geometric sampling
        self.delta = delta # allowed error on the estimated mean
        self.epsilon = epsilon # corresponding epsilon to the delta for epsilon-delta criterium
        self.p = p # scaling factor for d
        self.range = range # range of the random value
        self.t = 1 # t-th taken sample
        self.k = 0 # updating factor for geometric sampling
        self.alpha = 0 # factor in geometric sampling: fraction of samples from two following iterations
        self.x = 0 # variable in calculation
        self.c = self.delta*(self.p-1)/self.p # constant factor for d_t

        # values for tracking and updating the mean and variance
        self.error_values = []
        self.x_mean_array = [0] # predicted mean over i samples
        self.variance_array = [0] # variance of the mean over i samples
        
        # values for the Welford's algorithm
        self.mean = 0
        self.mean_sum = 0

        # track the values for the stopping decision
        self.ct_history = []

        
    # use Welford's algorithm to udpdate the mean and variance
    def update(self, sample):
        self.error_values.append(sample)
        if(len(self.error_values) >= 1):
            self.t = self.t + 1

            delta = sample - self.mean
            new_mean = self.mean + delta/self.t
            self.mean_sum = self.mean_sum + delta*(sample - new_mean)

            self.mean = new_mean
            self.x_mean_array.append(self.mean)
            self.variance_array.append(self.mean_sum/self.t)
    

    def algo(self, sample):
        # calculate the ith predicted mean and its variance
        self.update(sample)
        if(self.t > np.floor(self.beta**self.k)):
            self.k = self.k+1
            self.alpha = np.floor(self.beta**self.k)/np.floor(self.beta**(self.k-1))
            dk = self.c/(self.k**self.p) # d = series of probabilities going to delta
            self.x = -self.alpha*np.log(dk/3)
            ct = self.variance_array[-1]*np.sqrt(2*self.x/self.t) + 3*self.range*self.x/self.t
            self.ct_history.append(ct)
        
        print(ct)
        if self.k == 0:
            return True
        if self.ct_history[-1] > self.epsilon:
            return True
        else:
            return False
        
    def __call__(self, data_frame):
        data = data_frame.copy()
        keep_running = True

        # TODO extraction of error values of the batch in the dataframe
        # TODO handling of each configuration being able to stop individually
        for sample in data.groupby(self.config_columns):
            if(keep_running):
                keep_running = self.algo(sample)



class WilcoxonStop(Stopping):
    def __init__(self, config_columns, error, threshold=0.05, p_threshold=0.05):
        self.error = error
        self.config_columns = config_columns
        self.threshold = threshold
        self.p_threshold = p_threshold
        self.best_config = None
        self.configs = {}

    def compute_error(self, p_est, p_val):
        match self.error:
            case "mae":
                return mae(p_est, p_val)
            case "mrae":
                return mrae(p_est, p_val)
            case "mkld":
                return mkld(p_est, p_val)
        

    def __call__(self, dataframe):
        data = dataframe.copy()
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
            min_err[key] = np.mean([x[1] for x in values],)

        self.best_config = (min(min_err.items(), key=lambda x: x[1]))
        self.best_config = (self.best_config[0], [x[1] for x in errors[self.best_config[0]]])
        errors.pop(self.best_config[0], None)

        for configuration, values in errors.items():
            y = [x[1] for x in values]
            w_statistic, p_value = wilcoxon(x=self.best_config[1], y=y)
            if(p_value < self.p_threshold):
                mask = (
                    (data["quantifier"] == configuration[0]) &
                    (data["C"] == configuration[1]) &
                    (data["class_weight"] == configuration[2])
                )
                data.loc[mask, "stopped"] = True
                
        data = data.drop(columns=["val_error"])

        return data


class RankingStop(Stopping):
    def __init__(self, config_columns, num_iterations, error, number_equal_configs=0):
        self.config_columns = config_columns
        self.ranking_history = []
        self.num_iterations = num_iterations # number of iterations the ranking should stay the same before stopping
        self.counter = 0
        self.error = error
        self.number_equal_configs = number_equal_configs # set the number of considered configurations for stopping

    def compute_error(self, p_est, p_val):
        match self.error:
            case "mae":
                return mae(p_est, p_val)
            case "mrae":
                return mrae(p_est, p_val)
            case "mkld":
                return mkld(p_est, p_val)

    def __call__(self, dataframe):
        if(len(dataframe) == 0):
            raise ValueError("The passed dataframe is empty.")
        data = dataframe.copy()
        data = data[data["stopped"] == False] # consider only not stopped configurations

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

        if(np.all(self.ranking_history[:self.number_equal_configs] == config_rank[:self.number_equal_configs])):
            #self.counter = self.counter + dataframe["val_sample"].nunique()
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
    def __init__(self,dataset, batch_size, starting_index=0):
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

