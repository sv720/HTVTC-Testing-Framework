import optuna
from optuna.samplers import RandomSampler
import json
#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from optuna.samplers import RandomSampler
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator, crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

quantity = 'EXEC-TIME'

task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data, method = 'cross_validation')
func = crossValidationFunctionGenerator(data_split, algorithm='knn-classification', task=task)

timestamps = []
validation_losses = []
best_loss = None

def objective(trial):
    N = trial.suggest_int("N", 1, 101, step=1)
    p = trial.suggest_int("p", 1, 101, step=1)
    weightingFunction = trial.suggest_categorical("weightingFunction", ['uniform', 'distance'])
    distanceFunction = trial.suggest_categorical("distanceFunction", ['minkowski'])

    loss = func(N=N, weightingFunction=weightingFunction, distanceFunction=distanceFunction, p=p, metric=classificationmetrics.indicatorFunction)

    timestamp = time.perf_counter_ns()
    #Update best loss based on received loss value
    global best_loss
    if best_loss is None:
        best_loss = loss
    else:
        best_loss = min(best_loss, loss)
    #Store timestamp and loss
    timestamps.append(timestamp)
    validation_losses.append(best_loss)
    return loss

#Start timer/memory profiler/CPU timer
start_time = None
if quantity == 'EXEC-TIME':
    import time
    start_time = time.perf_counter_ns()
elif quantity == 'CPU-TIME':
    import time
    start_time = time.process_time_ns()
elif quantity == 'MAX-MEMORY':
    import tracemalloc
    tracemalloc.start()

optuna.logging.set_verbosity(optuna.logging.FATAL)
study = optuna.create_study(sampler=RandomSampler(seed=1))
study.optimize(objective, n_trials=80) #value in first report table: 5000

#resource_usage = getrusage(RUSAGE_SELF)
#End timer/memory profiler/CPU timer
result = None
if quantity == 'EXEC-TIME':
    end_time = time.perf_counter_ns()
    result = end_time - start_time
elif quantity == 'CPU-TIME':
    end_time = time.process_time_ns()
    result = end_time - start_time
elif quantity == 'MAX-MEMORY':
    _, result = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
print('\n\n\n')
print(f'Number of trials: {len(study.trials)}')
print(f'Best trial: {study.best_trial}')
print(f'{quantity}: {result}')
if quantity == 'EXEC-TIME':
    print(f'EXEC-TIME in s : {result * (10**(-9))}')
#print(f'Resource usage: {resource_usage}')

"""
#Process time stamps
for i in range(len(timestamps)):
    timestamps[i] -= start_time

graph_stats = {
    'time': timestamps,
    'loss': validation_losses
}
PATH = 'graphs/KNN-classification.json'
with open(PATH, 'w') as fp:
        json.dump(graph_stats , fp)

"""