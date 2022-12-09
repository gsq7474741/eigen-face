from nni.experiment import Experiment

if __name__ == '__main__':



    model_name = 'svm'

    if model_name == 'knn':
        search_space = {
            'energy': {'_type': 'uniform', '_value': [0.5, 1]},
            'hw': {'_type': 'randint', '_value': [20, 100]},
            'n_neighbors': {'_type': 'randint', '_value': [1, 10]},
            'model_name': {'_type': 'choice', '_value': ['knn']},
        }
    elif model_name == 'svm':
        search_space = {
            'energy': {'_type': 'uniform', '_value': [0.5, 1]},
            'hw': {'_type': 'randint', '_value': [20, 100]},
            'c': {'_type': 'uniform', '_value': [0, 5]},
            'gamma': {'_type': 'uniform', '_value': [0, 5]},
            'model_name': {'_type': 'choice', '_value': ['svm']},
        }
    elif model_name == 'mlp':
        search_space = {
            'energy': {'_type': 'uniform', '_value': [0.5, 1]},
            'hw': {'_type': 'randint', '_value': [20, 100]},
            'hidden_layer_sizes': {'_type': 'choice', '_value': [(100,), (200,), (300,)]},
            'learning_rate': {'_type': 'loguniform', '_value': [1, 1e-6]},
            'model_name': {'_type': 'choice', '_value': ['mlp']},
        }
    else:
        raise ValueError('model_name must be knn or svm or mlp')

    experiment = Experiment('local')
    experiment.config.experiment_name = f'orl_{model_name}'
    experiment.config.trial_command = 'python orl_main.py'
    experiment.config.trial_code_directory = '.'

    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    experiment.config.max_trial_number = 2000
    experiment.config.trial_concurrency = 16

    experiment.run(8081)
    input('Press enter to quit')
    experiment.stop()
    experiment.view('vuwpo3kt',63000)

# [2022-12-09 00:56:28] Creating experiment, Experiment ID: nzpu1t8x
# [2022-12-09 01:50:50] Creating experiment, Experiment ID: vuwpo3kt
