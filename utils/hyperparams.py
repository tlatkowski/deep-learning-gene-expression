import io

import yaml


class Hyperparameters:
  def __init__(self, path_to_config_file):
    with io.open(path_to_config_file) as file:
      config = yaml.load(file)
    self.learning_rate = config['hyperparameters']['learning_rate']
    self.input_size = config['hyperparameters']['input_size']
    self.hidden_sizes = config['hyperparameters']['hidden_sizes']
    self.output_size = config['hyperparameters']['output_size']
    self.num_features = config['hyperparameters']['num_features']
    self.activation_function = config['hyperparameters']['activation_function']
    # TODO add force feature selection
    # TODO gradient descent with momoentum
    # TODO decorators for plots
    self.num_layers = len(self.hidden_sizes) + 1
    self.selection_methods = config['selection_methods']
    
    self.num_epochs = config['training']['num_epochs']
    self.batch_size = config['training']['batch_size']  # online learning when batch_size=1
    self.cross_validation_folds = config['training']['cross_validation_folds']  # TODO when === num of observations then leave-one-out is applied.
    self.lambda_reg = 0.8
    self.norm_data = True
    # TODO plot cost with and without reguralization
    
    # dirs:
    self.data_file = 'data/data.tsv'
