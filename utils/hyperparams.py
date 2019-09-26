import io

import yaml


class Hyperparameters:
  def __init__(self, path_to_config_file):
    with io.open(path_to_config_file) as file:
      config = yaml.load(file)
    self.learning_rate = 0.001
    self.input_size = 100
    self.hidden_sizes = [80]
    self.output_size = 1
    self. num_features = 100
    self.activation_function = 'tanh'
    # TODO add force feature selection
    # TODO gradient descent with momoentum
    # TODO decorators for plots
    self.num_layers = len(self.hidden_sizes) + 1
    self.selection_methods = ['fisher', 'corr', 'ttest', 'random']
    
    self.num_epochs = 10000
    self.batch_size = 20  # online learning when batch_size=1
    self.cross_validation_folds = 10  # TODO when === num of observations then leave-one-out is applied.
    self.lambda_reg = 0.8
    self.norm_data = True
    # TODO plot cost with and without reguralization
    
    # dirs:
    self.data_file = 'data/data.tsv'
