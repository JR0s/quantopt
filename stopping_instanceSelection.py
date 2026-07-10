import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from abc import ABC, abstractmethod

MAX_INT = 10000000

class Stopping(ABC):
    @abstractmethod
    def __call__(self, sampled_value): # returnsstopping decision
        pass


class RandomStop(Stopping):
    def __init__(self, config_columns, n_samples):
        self.config_columns = config_columns
        self.n_samples = n_samples

    def __call__(self, dataframe):
        dataframe = dataframe.copy()
        dataframe["stopped"] = dataframe.groupby(self.config_columns)["val_sample"].transform(lambda gdf: gdf.nunique() >=  self.n_samples)
        return dataframe

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

    def get_results(self):
        pass


class WilcoxonStop(Stopping):
    def __init__(self, instances):
        self.instances = instances
        
    #
    # wilcoxon_history ->
    # y_sampled ->
    # threshold ->
    # ema ->
    # ema_val ->
    # v ->
    # p_val_thresh ->
    def algo():
        wilcoxon_history = []
        y_sampled = []
        threshold = 0
        p_val_thresh = 0
        x = []
        y = []
        ema = wilcoxon(x,y)
        if(len(wilcoxon_history) == 0 or np.count_nonzero(y_sampled)/y_sampled.shape[0] < threshold):
            return False
        else:
            ema_val = wilcoxon_history[0]
            for v in wilcoxon_history[1:]:
                ema_val = ema * v + (1 - ema) * ema_val
            return ema_val <= p_val_thresh


class RankingStop(Stopping):
    def __init__(self, instances):
        self.instances = instances

    #
    # ranking_history ->
    # con_num_instances ->
    # min_amount ->
    # y_sampled ->
    #
    #
    def algo():
        ranking_history = 0
        con_num_instances = 0
        min_amount = 0
        y_sampled = []
        if (len(ranking_history) < con_num_instances or
                np.count_nonzero(y_sampled) / y_sampled.shape[0] < min_amount):
            return False
        else:
            recent = ranking_history[-1]
            for i in range(2, con_num_instances + 1):
                if not np.all(recent == ranking_history[-i]):
                    return False
            return True
        

class TStop(Stopping):
    def __init__(self, instances):
        self.instances = instances

    def algo():
        pass




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

