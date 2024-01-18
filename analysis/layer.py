import numpy as np
import torch
from .casper import nethook
def trace_with_patch_layer(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect  
):
    prng = np.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    layers = [states_to_patch[0], states_to_patch[1]]

    # Create dictionary to store intermediate results
    inter_results = {}

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer not in layers:
            return x

        if layer == layers[0]:
            inter_results["hidden_states"] = x[0].cpu()
            inter_results["else"] = x[1]
            return x
        elif layer == layers[1]:
            short_cut_1 = inter_results["hidden_states"].cuda()
            short_cut = (short_cut_1, inter_results["else"])
            return short_cut
            
    with torch.no_grad(), nethook.TraceDict(
        model,
        layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs