import quapy as qp
import numpy as np
from sklearn.linear_model import LogisticRegression
import quapy.functional as F
from quapy.data.datasets import LEQUA2022_SAMPLE_SIZE, fetch_lequa2022
from quapy.evaluation import evaluation_report
from quapy.method.aggregative import SLD
from quapy.model_selection import GridSearchQ
import pandas as pd


task = 'T1B'

# set the sample size in the environment. The sample size is task-dendendent and can be consulted by doing:
qp.environ['SAMPLE_SIZE'] = LEQUA2022_SAMPLE_SIZE[task]
qp.environ['N_JOBS'] = -1

# the fetch method returns a training set (an instance of LabelledCollection) and two generators: one for the
# validation set and another for the test sets. These generators are both instances of classes that extend
# AbstractProtocol (i.e., classes that implement sampling generation procedures) and, in particular, are instances
# of SamplesFromDir, a protocol that simply iterates over pre-generated samples (those provided for the competition)
# stored in a directory.
training, val_generator, test_generator = fetch_lequa2022(task=task)
Xtr, ytr = training.Xy

# define the quantifier
quantifier = SLD(classifier=LogisticRegression(C=0.0177827941003892, class_weight=None, max_iter=1000))


trained_quantifier = quantifier.fit(Xtr, ytr)

# evaluation
report = evaluation_report(quantifier, protocol=test_generator, error_metrics=['mae', 'mrae', 'mkld'], verbose=True)

# printing results
pd.set_option('display.expand_frame_repr', False)
report['estim-prev'] = report['estim-prev'].map(F.strprev)
print(report)

report.to_csv("best_mkld.csv")

print('Averaged values:')
print(report.groupby(["true-prev", "estim-prev"]).mean())
