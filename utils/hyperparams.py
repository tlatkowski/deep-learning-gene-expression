class Hyperparameters(object):
    learning_rate = 0.001
    input_size = 100
    hidden_sizes = [80]
    output_size = 1
    num_features = 100
    activation_function = 'tanh'
    # TODO add force feature selection
    # TODO gradient descent with momoentum
    # TODO decorators for plots
    num_layers = len(hidden_sizes) + 1
    selection_methods = ['fisher', 'corr', 'ttest', 'random']

    num_epochs = 10000
    batch_size = 20  # online learning when batch_size=1
    cross_validation_folds = 10  # TODO when === num of observations then leave-one-out is applied.
    lambda_reg = 0.8
    norm_data = True
    # TODO plot cost with and without reguralization

    # dirs:
    data_file = 'data/data.tsv'
