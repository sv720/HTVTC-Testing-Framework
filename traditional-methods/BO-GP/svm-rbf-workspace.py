#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bayes_opt import BayesianOptimization
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)
data_split = trainTestSplit(data)
func = evaluationFunctionGenerator(data_split, algorithm='svm-rbf', task=task)


def objective(C, gamma):
    #subtract from 1 because the library only supports maximise
    return 1 - func(C, gamma, metric=classificationmetrics.indicatorFunction)

start_time = time.perf_counter()


pbounds = {'C': (0.05, 5.0), 'gamma': (0.05, 5.0)}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

#resource_usage = getrusage(RUSAGE_SELF)
end_time = time.perf_counter()
print('\n\n\n')
print(f'best combination: {optimizer.max}')
print(f'Execution time: {end_time - start_time}')
#print(f'Resource usage: {resource_usage}')
