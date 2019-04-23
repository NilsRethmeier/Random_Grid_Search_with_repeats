import itertools
import random
from collections import OrderedDict
import csv
import warnings
import os
import time
import datetime
from sqlalchemy import create_engine


class ParamConfigGen():

    def estimate_time(self):
        # self.num_experiments = num_experiments
        elapsed = time.time() - self.start_time
        remaining = (elapsed / self.experiments_count) * (self.num_experiments - self.experiments_count)
        return str(datetime.timedelta(seconds=remaining))

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
        >>> {'lr': 0.1, 'data_size': 1, 'max_ordinal_int': 3}
            {'lr': 0.5, 'data_size': 1, 'max_ordinal_int': 3}
            {'lr': 0.1, 'data_size': 1, 'max_ordinal_int': 4}
            {'lr': 0.5, 'data_size': 1, 'max_ordinal_int': 4}
            {'lr': 0.1, 'data_size': 2, 'max_ordinal_int': 3}
            {'lr': 0.5, 'data_size': 2, 'max_ordinal_int': 3}
            {'lr': 0.1, 'data_size': 2, 'max_ordinal_int': 4}
            {'lr': 0.5, 'data_size': 2, 'max_ordinal_int': 4}

        # Usage
        >>> pc = ParamConfigGen(result_storage_location='/home/USER/tmp', model_family='none')
        >>> print [x for x in pc.parameterConfigGenerator({'ngam': [2, 3], 'dropout': [0.1, 0.2], 'lr': [0.01, 0.02, 0.03]}, \
                                                     random_search=True, CV=2, max_experiments=5)]
        """
        i = 1
        for k in dicts:
            i *= len(dicts[k])
        num_experiments_to_run = i * CV + len(test_folds)  # to get data_configs * model_configs
        if verbose:
            print("Num param configs to test", num_experiments_to_run)
        config_iter = (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
        num_confs = 0
        self.start_time = time.time()
        self.experiments_count = 1
        if not random_search:
            for cv in range(CV):
                for fold in test_folds:
                    for conf in config_iter:
                        config = conf.copy()  # so that a list comprehension does not overwrite previous [CV] settings
                        config['CV_run'] = cv
                        config['fold'] = fold  # defailt should be verwritten
                        yield config
                        num_confs += 1
                        if num_confs % max_experiments == 0:
                            return
        else:
            combos = [x for x in config_iter]  # compute all combos
            num_experiments = len(combos) if max_experiments is None else max_experiments
            num_experiments = min(len(combos),
                                  max_experiments)  # in case max_experiments exceeds the num of generated configurations
            if verbose:
                print("Num Exp:", num_experiments)
                list1 += [num_experiments]
            self.num_experiments = num_experiments
            samples = random.sample(combos, num_experiments)  # sample a subset
            random.shuffle(samples)  # shuffle the order
            for sample in samples:  # generate num_experiments samples
                for fold in test_folds:
                    for cv in range(CV):  # add CV run info
                        config = sample.copy()  # so that a list comprehension does not overwrite previous [CV] settings
                        config['CV_run'] = cv
                        config['fold'] = fold  # defailt should be verwritten
                        yield config
                        self.experiments_count += 1


def test_random_cv_gen():
    params = OrderedDict(
        [('channels', [{'topic': {"type": "pre-trained", "tune": True}, 'w2v': {"type": "pre-trained", "tune": True}}]),
         ("batch_size", [64, 256]),
         ("max_worse_itterations", [35, 70]),
         ("epochs", [21]),
         ("t_fc_dim", [512, 2048]), ])

    sample = [x for x in parameterConfigGenerator(list1,params, random_search=False, CV=2, max_experiments=5)]
    # print(sample, len(sample))
    return sample


# test_random_cv_gen()

class DBwriter():

    def __init__(self, username, password, db_name, port=5432, host='localhost'):  # postgres data
        self.db_login = 'postgresql://' + username + ':' + password + "@" + host + ":" + str(port) + "/" + db_name
        self.str_to_numpy_array = lambda x: np.fromstring(
            x.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=' ')

    def store_in_db(dataframe, table='sentence_embeddings'):
        # postgres data
        engine = create_engine(self.db_login)
        dataframe.to_sql(name=table, con=engine, if_exists='append', index=False)
        engine.dispose()  # no leaks of time outs please

    def read_experiment_results_from_db(dataframe, table='sentence_embeddings', experiment_name=None,
                                        cols_to_numpy=['preds']):
        engine = create_engine(self.db_login)
        df_sql = pd.read_sql_query("SELECT * FROM \"" + table + "\" Where labels=" + "\'" + experiment_name + "\';",
                                   con=engine)
        for col in cols_to_numpy:
            df_sql[col] = df_sql[col].apply(self.str_to_numpy_array)
        engine.dispose()
        return df_sql


class ReComputationGuard():

    def __init__(self, path, params, fields_to_ignore_in_hashing=[], finished_fields=None):
        """
        Scans output file for parameter configs that were already successfully evaluated.
        :param path: path the the output csv (scans for params if already exists)
        :param data_params: data dependent parameters
        :param model_params: model params
        :param fields_to_ignore_in_hashing: some fields change (e.g. model type carries memory address -> changes)
        :return: instance
        """
        # store which fields to ignore in hashing
        self.ignore_fields = fields_to_ignore_in_hashing
        # compute the params that have already been processed
        new_header = sorted(list(set(params.keys())) + fields_to_ignore_in_hashing + finished_fields)
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            # never created, or empty
            self.already_trained_params = None
            self.old_header = new_header  # set the new header
        else:
            self.hash_order = sorted(params.keys())
            training_finshed_elements = set()
            if finished_fields is not None:
                training_finished_fields = finished_fields
            else:
                # example of fields that indicate if the run was finished or errored
                training_finished_fields = 'test_loss', 'test_acc', 'val_acc', 'val_loss', 'train_acc', 'train_loss', \
                                           'test_AUC', 'test_F1', 'model_storage_location'
            with open(path, 'r') as f:
                csv_reader = csv.DictReader(f, dialect="excel", delimiter="\t")
                self.old_header = csv_reader.fieldnames
                # NOT WORKING: since not all measures are known at file creation
                if set(self.old_header) != set(new_header):  # existing files header and new header should be the same
                    print("OLD", sorted(self.old_header))
                    print("NEW", sorted(new_header))
                    print("DIF", sorted(set(self.old_header).symmetric_difference(set(new_header))))
                    raise Exception("Incompatible CSV headers")
                for line in csv_reader:
                    # only add to list of finished experiments if all scores are present (no sub 100% experiments)
                    if not any([line[key] is None for key in training_finished_fields]):
                        # record existing parameter info
                        training_finshed_elements.add(self.make_hash(line))
            self.already_trained_params = training_finshed_elements

    def make_hash(self, dikt):
        """
        Turns parameter name:value pairs into a hashable string
        :param dikt: the parameter name:value dict
        :param fields_to_ignore: model_variant is the address of an instance (always changes, irrelevant for comparison)
        :return:
        """
        hashable_parts = ["%s:%s" % (param, str(dikt[param])) for param in self.hash_order if
                          param not in self.ignore_fields]
        return "\t".join(hashable_parts)

    def experiment_already_run(self, dikt):
        """
        Check if we already have result for this parameter config. if so we can skip computation.
        :param dikt:
        :return:
        """
        if self.already_trained_params is None:
            return False
        current_config_hash = self.make_hash(dikt)
        current_config_hash = current_config_hash.replace("None", "")
        return current_config_hash in self.already_trained_params

    def get_header(self):
        return self.old_header


class RecomputationGuardedCSVDictWriter():
    """ Reads a csv and checks it once (initially) for already run configuartions """

    def __init__(self, path, params, fields_to_ignore_in_hashing=[], finished_fields=None, dialect="excel",
                 delimiter="\t"):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.file = open(path, 'a+')
        self.recomp_guard = ReComputationGuard(path, params, fields_to_ignore_in_hashing, finished_fields)
        self.tsv_writer = csv.DictWriter(self.file, self.recomp_guard.get_header(), dialect=dialect,
                                         delimiter=delimiter)

        if self.recomp_guard.already_trained_params == None:
            # write the csv header only once (when file is new)
            self.tsv_writer.writeheader()
        self.DB = None

    def init_DBWriter(username=None, password=None, db_name=None, port=5432, host='localhost'):
        self.DB = DBwriter(username, password, db_name, port, host)

    def check_if_already_run(self, run_params, verbose=False):
        already_run = self.recomp_guard.experiment_already_run(run_params)
        if verbose and already_run:
            print("#### already trained", run_params)
        return already_run

    def writerow(self, result_dict):
        """ """
        if not self.recomp_guard.experiment_already_run(result_dict):
            self.tsv_writer.writerow(result_dict)
            self.file.flush()
        else:
            warnings.warn("Experiment already run, not writing results: Conflicting config=" + str(result_dict),
                          RuntimeWarning)

    def store_in_db(dataframe, table='sentence_embeddings'):
        if self.DB is not None:
            self.DB.store_in_db(dataframe, table)
        else:
            raise Exception('use .init_DBWriter() to init db first')

    def store_in_db(dataframe, table='sentence_embeddings', experiment_name=None, cols_to_numpy=['preds']):
        if self.DB is not None:
            self.DB.store_in_db(dataframe, table)
        else:
            raise Exception('use .init_DBWriter() to init db first')

    def read_experiment_results_from_db(dataframe, table='sentence_embeddings', experiment_name=None,
                                        cols_to_numpy=['preds']):
        return self.DB.read_experiment_results_from_db(dataframe, table, experiment_name, cols_to_numpy)

    def close(self):
        self.file.close()


def test_RecomputationGuardedCSVDictWriter():
    for i in range(2):
        params = test_random_cv_gen()
        params[0]['mse_loss'] = 1.0
        print(params[0])
        guarded_csv = RecomputationGuardedCSVDictWriter(path='results/test.csv',
                                                        params=params[0],
                                                        fields_to_ignore_in_hashing=[],
                                                        finished_fields=['mse_loss'])
        guarded_csv.writerow(params[0])
        print(guarded_csv.check_if_already_run(params[0]))
        params[1]['mse_loss'] = 0.9
        print(guarded_csv.check_if_already_run(params[1]))
        params[2]['mse_loss'] = 0.8
        print(guarded_csv.check_if_already_run(params[2]))
        guarded_csv.writerow(params[2])
        guarded_csv.close()
        print('RUN:', i)
        # should display warnings
