# CASPER: Causality Analysis for Evaluating the Security of Large Language Models

This is the code repository for the CASPER paper on applying causality analysis to evaluate the security of large language models (LLMs).

## Abstract

Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.

## Code Structure

The code is organized as follows:

- `casper/`: Implementation of the CASPER causality analysis framework
- `analysis/`: Code for layer-based analysis and neuron-based analysis
- `models/`: LLMs
- `attacks/`: Adversarial attack methods informed by CASPER
- `experiments/`: Scripts to reproduce key results on discovering vulnerabilities
- `utils/`: Utility functions for processing data

## Setup

The code was developed with Python 3.7. To install dependencies:
```bash
pip install -r requirements.txt
```

## Model
Please follow the instructions to download Vicuna-7B or/and LLaMA-2-7B-Chat first (we use the weights converted by HuggingFace [here](https://huggingface.co/meta-llama/Llama-2-7b-hf)).

## Demo
We include a notebook `demo.ipynb` which provides an example on analysis on Llama2-7B

Some experiments results are shown in https://casperllm.github.io/

## Reproducibility

- A note for hardware: all experiments we run use one or multiple NVIDIA A100 GPUs, which have 80G memory per chip. 

- You can directly run the demo.ipynb with the llama2 or vicuna model. If you attempt to run with other chat model, ensure to load the correct conversation template from fastchat.

- Ensure the transformer version is correct, if fail to run demo.ipynb for layer and neuron analysis, you should try the code in the analysis folder.

## Citation
If you find this useful in your research, please consider citing:

``````
@article{zhao2023causality,
  title={Causality Analysis for Evaluating the Security of Large Language Models},
  author={Zhao, Wei and Li, Zhe and Sun, Jun},
  journal={arXiv preprint arXiv:2312.07876},
  year={2023}
}
