import io
import logging

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from utils import features_utils, nn_utils
from utils.data_params import DataParams as dp
from utils.hyperparams import Hyperparameters as hp

EXPERIMENT_CONFIG_FILE_PATH = 'config/experiment_setup.yml'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Y_true = np.reshape(np.array([1] * 82 + [0] * 64), (1, 146))


def load_config():
  with io.open(EXPERIMENT_CONFIG_FILE_PATH) as file:
    config = yaml.load(file)
  return config


def main():
  config = load_config()
  logger.info('Loading data...')
  df = pd.read_csv(hp.data_file, sep='\t', header=None, index_col=0).T
  logger.info('Loaded data...')
  
  shuffle = np.random.permutation(Y_true.shape[1])
  X = nn_utils.norm_data(df)
  X['Case'] = ['AUTISM'] * dp.num_autism + ['CONTROL'] * dp.num_control
  
  skf = StratifiedKFold(n_splits=hp.cross_validation_folds)
  cv_acc = {'fisher': [], 'corr': [], 'ttest': [], 'random': []}
  
  for fold_id, (train_idxs, test_idxs) in enumerate(skf.split(X.values, Y_true.reshape(146))):
    X_train = X.iloc[train_idxs]
    Y_train = Y_true[:, train_idxs]
    X_test = X.iloc[test_idxs]
    Y_test = Y_true[:, test_idxs]
    
    selection_methods = config['selection_methods']
    selected_features = features_utils.execute_selection(selection_methods, X_train)
    for method, X_train_sel_features in selected_features.items():
      init_parameters = nn_utils.init_parameters(input_size=hp.input_size,
                                                 hidden_sizes=hp.hidden_sizes,
                                                 output_size=hp.output_size)
      trained_params, _ = nn_utils.train_nn(X_train_sel_features,
                                            Y_train,
                                            init_parameters,
                                            method,
                                            hp.activation_function,
                                            '[{}/{}]'.format(fold_id + 1,
                                                             hp.cross_validation_folds))
      
      X_test_sel_features = features_utils.apply_selection(method, X_test,
                                                           num_features=hp.num_features)
      fold_acc = nn_utils.test_nn(X_test_sel_features, Y_test, trained_params, method,
                                  hp.activation_function)
      cv_acc[method].append(fold_acc)
    
    for m in hp.selection_methods:
      logger.info('%d-fold cross-validation accuracy for [%s] method : [%d]',
                  hp.cross_validation_folds, m, sum(cv_acc[m]) / len(cv_acc[m]))


if __name__ == '__main__':
  main()
