#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.examples.commons import MyWorker
from hpbandster.optimizers import BOHB as BOHB

from commonfunctions import generate_range
from trainmodels import crossValidationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics

#To hide logs
import logging
logObj = logging.getLogger('noOutput')
logObj.setLevel(100)

#To hide warnings
import warnings
warnings.filterwarnings("ignore")


task = 'regression'
data = loadData(source='sklearn', identifier='california_housing', task=task)
metric = regressionmetrics.mae

#Define the worker
class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sleep_interval = sleep_interval

    def compute(self, config, budget, **kwargs):
        data_split = trainTestSplit(data, method='cross_validation')
        func = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task, budget_type='samples', budget_fraction=budget)
        res = func(**config, metric=metric)
        
        return({
                    'loss': res,
                    'info': res
                })
    
    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        N = CSH.UniformIntegerHyperparameter('N', lower=1, upper=100)
        p = CSH.UniformIntegerHyperparameter('p', lower=1, upper=100)
        cs.add_hyperparameters([N, p])

        weightingFunction = CSH.CategoricalHyperparameter('weightingFunction', ['uniform', 'distance'])
        distanceFunction = CSH.CategoricalHyperparameter('distanceFunction', ['minkowski'])
        cs.add_hyperparameters([weightingFunction, distanceFunction])

        return cs

#Setup nameserver
NS = hpns.NameServer(run_id='knn-r', host='127.0.0.1', port=None)
NS.start()

#Start a worker
w = MyWorker(sleep_interval = 0, nameserver='127.0.0.1',run_id='knn-r', logger=logObj)
w.run(background=True)

quantity = 'EXEC-TIME'

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

#Run the optimiser
MAX_BUDGET = 1.0
MIN_BUDGET = 0.2
bohb = BOHB(  configspace = w.get_configspace(),
              run_id = 'knn-r', nameserver='127.0.0.1',
              min_budget=MIN_BUDGET, max_budget=MAX_BUDGET,
              logger=logObj
           )
res = bohb.run(n_iterations=50)

#End timer/memory profiler/CPU timer
quantity_result = None
if quantity == 'EXEC-TIME':
    end_time = time.perf_counter_ns()
    quantity_result = end_time - start_time
elif quantity == 'CPU-TIME':
    end_time = time.process_time_ns()
    quantity_result = end_time - start_time
elif quantity == 'MAX-MEMORY':
    _, quantity_result = tracemalloc.get_traced_memory()
    tracemalloc.stop()

#Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

id2config = res.get_id2config_mapping()
inc_id = res.get_incumbent_id()
inc_runs = res.get_runs_by_id(inc_id)
inc_run = inc_runs[-1]

best_config = id2config[inc_id]['config']

print('Best found configuration:', best_config)
print(f'Validation loss: {inc_run.loss}')
print('A total of %i unique configurations were sampled.' % len(id2config.keys()))
print('A total of %i runs were executed.' % len(res.get_all_runs()))
print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/MAX_BUDGET))
print(f'{quantity}: {quantity_result}')

#Reevaluate to make sure the validation loss is correct
data_split = trainTestSplit(data, method='cross_validation')
revalfunc = crossValidationFunctionGenerator(data_split, algorithm='knn-regression', task=task)
reval = revalfunc(**best_config, metric=metric)
print(f'reevaluated loss: {reval}')
