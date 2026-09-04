import quapy as qp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from quapy.data.datasets import LEQUA2022_SAMPLE_SIZE, fetch_lequa2022
import pandas as pd
import itertools

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

x = []
y_true = []
for (X_i, y_i) in enumerate(val_generator()):
    x.append(X_i)
    y_true.append(y_i)

# define the list of classifiers
c = np.geomspace(1e-3, 1e2, 21)
w = [None, "balanced"]

classifier_params = list(itertools.product(c, w))

reports = {} 
for params in classifier_params:
    model_C, class_we = classifier_params
    classifier = LogisticRegression(C=model_C, class_weight=class_we, max_iter=1000)
    trained_classifier = classifier.fit(Xtr, ytr)
    y_pred = trained_classifier.predict(X=x)

    # evaluation
    report = classification_report(y_true=y_true, y_pred=y_pred)
    print(report)
    reports.extend(report)

print(reports)

df = pd.DataFrame(reports)
df.to_csv("classification.csv")