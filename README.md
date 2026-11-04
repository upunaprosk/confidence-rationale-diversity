# On the Impact of Instruction Tuning on Confidence and Lexical Diversity

<p align="center">
  <a href="https://aclanthology.org/2026.uncertainlp-main.33/">
    <img alt="ACL Anthology" src="https://img.shields.io/badge/ACL%20Anthology-Paper-2d6a9f">
  </a>
  <a href="https://arxiv.org/abs/2608.13430">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2608.13430-b31b1b.svg">
  </a>
  <a href="https://huggingface.co/papers/2608.13430">
    <img alt="Hugging Face" src="https://img.shields.io/badge/🤗-Hugging%20Face%20Paper-FFD21E">
  </a>
</p>

This repository contains code for the paper:

**Are You Sure You’re Sure? On the Impact of Instruction Tuning on Confidence and Lexical Diversity**   
Authors: Irina Proskurina, Mayank Kumar, and Oyindolapo Olabisi Komolafe   
> Third Workshop on Uncertainty-Aware NLP, EMNLP 2026


Abstract: 

>Instruction-tuned language models achieve strong performance across a range of generation tasks but have recently been shown to exhibit verbalized overconfidence, which may manifest in less diverse supporting rationales for incorrect answers. However, whether such overconfidence is associated with rationale consistency remains an open question. In this paper, we study whether changes in the lexical diversity of generated answer rationales accompany changes in model confidence induced by instruction tuning. We evaluate three matched base and instruction-tuned models across question-answering benchmarks and find that instruction tuning consistently increases answer confidence, despite limited changes in predictive accuracy, while degrading likelihood-based calibration. Secondly, we observe a non-uniform effect of instruction tuning on rationale diversity: cross-rationale diversity consistently decreases, whereas surface-level lexical diversity varies in both direction and magnitude across models and benchmarks. Finally, we find that these differences persist after controlling for answer selection and rationale length, confirming that confidence and rationale diversity capture distinct effects of instruction tuning.
>


The code contains the following main scripts:

- `evaluate.py` - likelihood-based confidence, verbalized confidence, and calibration evaluation; rationale generation and lexical diversity evaluation.
- `run_experiments.py` - Base and Instruct experiments across the supported models and datasets.
- `run_controls.py` - the controlled CommonsenseQA experiment for the Base/Instruct paired models.

## Installation

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

## Experiments

```
python run_experiments.py   --family all   --variant both   --task all   --k 5   --max-new-tokens 100   --results-root results/main
```
The full evaluation set is used by default.

For a small test use:
```
python run_experiments.py   --family qwen   --variant both   --task arc_easy   --n-items 20   --generation-items 20   --k 2   --results-root results/smoke
```

## Controlled experiment

```
python run_controls.py   --family all   --results-root results/main   --control-root results/control_csqa
```


## Citation

```
@inproceedings{proskurina-etal-2026-sure,
    title = "Are You Sure You{'}re Sure? On the Impact of Instruction Tuning on Confidence and Lexical Diversity",
    author = "Proskurina, Irina  and
      Kumar, Mayank  and
      Olabisi Komolafe, Oyindolapo",
    editor = {Aziz, Wilker  and
      Berant, Jonathan  and
      Eikema, Bryan  and
      Ilia, Evgenia  and
      de Marneffe, Marie-Catherine  and
      Plank, Barbara  and
      Shelmanov, Artem  and
      Swayamdipta, Swabha  and
      Tiedemann, J{\"o}rg  and
      Vazhentsev, Artem  and
      V{\'a}zquez, Ra{\'u}l  and
      Zerva, Chrysoula},
    booktitle = "The Proceedings of the Third Workshop on Uncertainty-Aware {NLP}",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.uncertainlp-main.33/",
    doi = "10.18653/2026.uncertainlp-main.33",
    pages = "544--554",
    ISBN = "979-8-89176-486-6",
    abstract = "Instruction-tuned language models achieve strong performance across a range of generation tasks but have recently been shown to exhibit verbalized overconfidence, which may manifest in less diverse supporting rationales for incorrect answers. However, whether such overconfidence is associated with rationale consistency remains an open question. In this paper, we study whether changes in the lexical diversity of generated answer rationales accompany changes in model confidence induced by instruction tuning. We evaluate three matched base and instruction-tuned models across question-answering benchmarks and find that instruction tuning consistently increases answer confidence, despite limited changes in predictive accuracy, while degrading likelihood-based calibration. Secondly, we observe a non-uniform effect of instruction tuning on rationale diversity: cross-rationale diversity consistently decreases, whereas surface-level lexical diversity varies in both direction and magnitude across models and benchmarks. Finally, we find that these differences persist after controlling for answer selection and rationale length, confirming that confidence and rationale diversity capture distinct effects of instruction tuning."
}
```
