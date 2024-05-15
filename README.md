# CoSPLADE: Contextualizing SPLADE for Conversational Information Retrieval

[![paper](https://img.shields.io/badge/arxiv-arXiv%3A2107.05720-brightgreen)](https://arxiv.org/abs/2301.04413)

This repository contains the source code for the CoSPLADE paper implemented with the [experimaestro-ir (XPMIR)](https://github.com/experimaestro/experimaestro-ir) library.

**The code is currently under development**: the first stage training is working.

To run the experiments on a cluster, you need to configure experimaestro [launchers](https://experimaestro-python.readthedocs.io/en/latest/launchers/) and [main settings](https://experimaestro-python.readthedocs.io/en/latest/configuration/).

## Installation

To install the necessary requirements, use

```sh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Training first-stage ranker

```sh
experimaestro run-experiment first-stage/normal
```


# cite

Please cite our work as:

```
@inproceedings{DBLP:conf/ecir/HaiGFNPS23,
  author    = {Nam Le Hai and
               Thomas Gerald and
               Thibault Formal and
               Jian{-}Yun Nie and
               Benjamin Piwowarski and
               Laure Soulier},
  editor    = {Jaap Kamps and
               Lorraine Goeuriot and
               Fabio Crestani and
               Maria Maistro and
               Hideo Joho and
               Brian Davis and
               Cathal Gurrin and
               Udo Kruschwitz and
               Annalina Caputo},
  title     = {CoSPLADE: Contextualizing {SPLADE} for Conversational Information
               Retrieval},
  booktitle = {Advances in Information Retrieval - 45th European Conference on Information
               Retrieval, {ECIR} 2023, Dublin, Ireland, April 2-6, 2023, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13980},
  pages     = {537--552},
  publisher = {Springer},
  year      = {2023},
  doi       = {10.1007/978-3-031-28244-7_34},
  biburl    = {https://dblp.org/rec/conf/ecir/HaiGFNPS23.bib},
}
```
