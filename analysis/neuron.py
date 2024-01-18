import numpy as np
import torch
from .casper import nethook

def trace_with_patch_neuron(
    model,  # The model
    inp,  # A set of inputs
    layers,  # what layer to perform causlity analysis
    neuron_zone, # zone of neurons
    answers_t,  # Answer probabilities to collect  
):

    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    layer = layers[0]
    start_neuron =  neuron_zone[0]
    end_neuron = neuron_zone[1]
    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer != layer:
            return x

        if layer == layer:
            h = untuple(x)
            zeros = torch.zeros_like(h)
            h[:, :, start_neuron:end_neuron] =  zeros[:, :, start_neuron:end_neuron] 
            x_2 = x[1]
            result = (h, x_2)
           
            return result

    with torch.no_grad(), nethook.TraceDict(
        model,
        layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)
    
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs