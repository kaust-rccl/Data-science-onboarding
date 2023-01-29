from dask.distributed import Client
import time
import os
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd

# get the number of cores with os.cpu_count()
n_cores = os.cpu_count()
print('Number of cores: %d' % n_cores, end="\n")

# #create a client and a local cluster
client = Client(processes=False, threads_per_worker=64,
                n_workers=n_cores, memory_limit='8GB')


param_grid = {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
              "kernel": ['rbf', 'poly', 'sigmoid'],
              "shrinking": [True, False]}

def main():
    grid_search = GridSearchCV(SVC(gamma='auto', random_state=0, probability=True),
                           param_grid=param_grid,
                           return_train_score=False,
                           cv=3,
                           n_jobs=-1)

    X, y = make_classification(n_samples=1000, random_state=0)

# here is the normal what to run grid search
    start = time.time()
    grid_search.fit(X, y)
    end = time.time()
    print ( 'CPU time: %f seconds' % (end - start) )



# here is the dask way to run grid search
    import joblib

    with joblib.parallel_backend('dask'):
        start = time.time()
        grid_search.fit(X, y)
        end = time.time()
        print ( 'Dask CPU time: %f seconds' % (end - start), end="\n" )
    print ( "This are metrics for the search\n{}".format (pd.DataFrame(grid_search.cv_results_).head() ) ,end = "\n" )

    print (f"best score for gridsreach : {grid_search.score(X, y)}" )


if __name__ == '__main__':
    main()