import numpy as np
from keras import backend as K

def get_activations(model, images, layers=None): 
    """For each image in emages get activation from layers in the model"""
    inp = model.input # input placeholder
    if layers is None:
        layers = [l.name for l in model.layers]
    outputs = [layer.output for layer in model.layers if (layer.name in layers) \
               and not (layer.name.startswith('drop'))]      # some layer outputs
    layers_names = [layer.name for layer in model.layers if layer.name in layers \
               and not (layer.name.startswith('drop'))]   # names for the output
    functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
    layer_outs = [func([images])[0] for func in functors]       # get activations for the batch
    return dict(zip(layers_names, layer_outs))

def model_receptive_fields(model, input_dim):
    """Find receptive fields for units in different layers of the model.
    Works only with the padding 'valid' (no padding) or padding 'same', when stride = 1"""
    
    cum_rf = {}
    last_rf = 1

    for l in model.get_config():
        l_type = l['class_name']
        l_name = l['config']['name']
        
        padding = None
        stride = None

        if l_type == 'Conv1D':
            kernel = l['config']['kernel_size'][0]
            stride = l['config']['strides'][0]
            if stride != 1:
                raise AssertionError("Function doesn't work for strides != 1.")
            padding = 0 if l['config']['padding'] == 'valid' else 1
            # TO DO: fix computation of padding size

        elif l_type == 'MaxPooling1D':
            kernel = l['config']['pool_size'][0]
            stride = l['config']['strides'][0]
            padding = 0 if l['config']['padding'] == 'valid' else 1
            
        elif l_type == 'Dense':
            last_rf = input_dim
            kernel = 1
        else:
            continue

        last_rf *= kernel
        cum_rf[l_name] = {'rf': last_rf, 'ks': kernel, 'p': padding, 's': stride}
        
    return cum_rf


def unit_act_input_idx(unit_n, padding, kernel_size, stride):
    start = unit_n - padding + stride - 1
    stop = unit_n - padding + kernel_size + stride -1
    return (start, stop)


def unit_act_to_input(unit_n, activation, input_shape, p, ks, s):
    """Find a segment in one-dimentional input, which activated particular neuron. 
    Map given activation onto the segment"""
    act = np.zeros(input_shape)
    start, stop = unit_act_input_idx(unit_n, p, ks, s)
    start = max(0, start)
    stop = min(stop, input_shape)
    act[np.arange(start, stop)] = activation
    return act

def filter_act_to_input(act_vector, model, layer, input_dim):
    """Activation of filter in respond to the input,
    (projects activations of all neurons in the activation map onto 1D-input)"""
    model_rfs = model_receptive_fields(model, input_dim)
    p = model_rfs[layer]['p']
    ks = model_rfs[layer]['ks']
    s = model_rfs[layer]['s']
    acts = np.vstack([unit_act_to_input(n, x, input_dim, p, ks, s) for (n, x) in enumerate(act_vector)])
    acts = np.sum(acts, axis=0)
    return acts