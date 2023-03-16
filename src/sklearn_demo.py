from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve ,auc

import os

cpu_per_node = os.environ['SLURM_JOB_CPUS_PER_NODE']



# create data points
X, y = make_classification(n_samples=100000, n_features=20, n_informative=4, random_state=0)

# split into trainin and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# train the model
model = RandomForestClassifier(n_estimators=int(cpu_per_node), max_depth=4, random_state=0, n_jobs=-1, verbose=2)
model.fit(X_train, y_train)

# get some results
y_hat = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_hat)

# print some results
print (f'ROC Curve (area = {auc(fpr, tpr)})')