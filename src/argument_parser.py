import os
from .models.generic_model import framework_list
from submodules.global_dl.training.optimizers import optimizer_list


training_arguments_das = [
    ['namespace', 'temperature', [
        [int, 'init_value', 5, 'initial value of the temperature parameters for to control Gumbel Softmax'],
        [float, 'decay_rate', 0.956, 'decay rate to anneal temperature'],
        [int, 'decay_steps', 1, 'decay steps']
    ]],

    ['namespace', 'optimizer', [
        [str, 'name', 'adam', 'optimize to be used for updating architecture parameters for search', lambda x: x.lower() in optimizer_list],
        [float, "lr", 0.1, 'learning rate', lambda x: x > 0],
        [float, 'momentum', 0.9, 'used when optimizer name is specified as sgd_momentum'],
    ]],
    [str, 'exported_architecture', 'export.yml', 'file to write the exported architecture'],
    [float, 'weight_decay', 1e-4, 'weight decay rate'],
    [float, 'arch_param_decay', 5e-4, 'weight decay rate for architecture parameters'],
    [int, 'num_warmup', 10, 'number of warmup epoch']
]
