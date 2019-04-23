# Random_Grid_Search_with_repeats
Code to do 1) random hyper parameter gridsearch with repetitions (reruns) in pytorch, 2) log training to visdom in a tensorboard like fashion, 3) generate an overall hyperparameter TSV file for exploration

Run_experiment.py contains a pseudo code example of how to use the
```Python
def parameterConfigGenerator(self, list1, dicts, random_search=False, CV=1, max_experiments=None, test_folds=[-1],
                             verbose=False):
    """
    Given a dict (ordered dict of parameters) this generates a Gridsearch/ Random Sample Grid Search
    :param dicts: Grid search dict/Ordered dict
    :param random: Uniformly sample from grid search
    :param CV: CV=2 a double random search is the best algorithm for finding an optimal model
    :param max_experiments: maximum number of experiment, None, means run all but in random order
    :param eval_folds: which fold to use as test set
    :return:
    Take a dictionary with parameters and their different configurations. Then make to cross product of all configurations.

    E.g.
    # with an orderedDict evaluated back to front, first
    >>> d = OrderedDict([("data_size", [1, 2]), ("max_ordinal_int", [3, 4]), ("lr", [.1, 0.5])])
    

    # Usage
    >>> pc = ParamConfigGen(result_storage_location='/home/tmp', model_family='none')
    >>> print [x for x in pc.parameterConfigGenerator({'ngam': [2, 3], 'dropout': [0.1, 0.2], 'lr': [0.01, 0.02, 0.03]}, \
                                                 random_search=True, CV=2, max_experiments=5)]
    >>> {'lr': 0.1, 'data_size': 1, 'max_ordinal_int': 3}
        {'lr': 0.5, 'data_size': 1, 'max_ordinal_int': 3}
        {'lr': 0.1, 'data_size': 1, 'max_ordinal_int': 4}
        {'lr': 0.5, 'data_size': 1, 'max_ordinal_int': 4}
        {'lr': 0.1, 'data_size': 2, 'max_ordinal_int': 3}
        {'lr': 0.5, 'data_size': 2, 'max_ordinal_int': 3}
        {'lr': 0.1, 'data_size': 2, 'max_ordinal_int': 4}
        {'lr': 0.5, 'data_size': 2, 'max_ordinal_int': 4}
    """
   ```
