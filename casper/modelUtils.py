import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import nethook

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        device = 'cuda:0'
    ):
        # print('wtf')
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            )
            nethook.set_requires_grad(False, model)
            model.eval().to(device)
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )
    
def generate_outputs(current_test_cases , mt, device='cuda:0',batch_size=1, max_new_tokens=100, verbose=True):
    """
    :param test_cases: a dictionary of test cases, where the keys are target behaviors
    :param model: the model to use for generation
    :param tokenizer: the tokenizer for the model
    :param max_new_tokens: the maximum size of each generation
    :return: a dictionary of generations, where the keys are target behaviors and the values are lists of generations
    """

    input_ids = mt.tokenizer(current_test_cases, padding=True, return_tensors="pt")
    input_ids['input_ids'] = input_ids['input_ids'].to(device)
    # np.save('input_ids_sec',input_ids['input_ids'].cpu().numpy())
    input_ids['attention_mask'] = input_ids['attention_mask'].to(device)
    num_input_tokens = input_ids['input_ids'].shape[1]
    # print(tokenizer.decode(input_ids['input_ids'].squeeze(0)))
    outputs = mt.model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'].half(),
                                     max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=mt.tokenizer.pad_token_id)
    generation = mt.tokenizer.batch_decode(outputs[:, num_input_tokens:], skip_special_tokens=True)
    # print(generation)
            
    return generation


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )

def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def predict_token(mt, prompts, return_p=False,device='cuda:0'):
    inp = make_inputs(mt.tokenizer, prompts,device=device)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p