[![Python Versions](https://img.shields.io/pypi/pyversions/gbiz_torch.svg)](https://pypi.org/project/gbiz_torch)
[![TensorFlow Versions](https://img.shields.io/badge/PyTorch-1.12+-blue.svg)](https://pypi.org/project/gbiz_torch)
[![Downloads](https://pepy.tech/badge/gbiz_torch)](https://pepy.tech/project/gbiz_torch)
[![PyPI Version](https://img.shields.io/pypi/v/deepctr.svg)](https://pypi.org/project/gbiz_torch)
[![GitHub Issues](https://img.shields.io/github/issues/whw199833/gbiz_torch.svg
)](https://github.com/whw199833/gbiz_torch/issues)
[![DOI](https://zenodo.org/badge/724196606.svg)](https://zenodo.org/doi/10.5281/zenodo.10222798)
[![License](https://img.shields.io/github/license/whw199833/gbiz_torch.svg)](https://github.com/whw199833/gbiz_torch/blob/master/LICENSE)
[![Activity](https://img.shields.io/github/last-commit/whw199833/gbiz_torch.svg)](https://github.com/whw199833/gbiz_torch/commits/master)

## Introduction to gbiz_torch

A comprehensive toolkit designed to help you accurately predict key metrics such as Click-Through Rates (CTR), Conversion Rates (CVR), uplift, and pricing strategies. Built with state-of-the-art algorithms and user-friendly interfaces, our package streamlines the process of forecasting and decision-making, allowing you to make data-driven choices with confidence. Whether you're looking to optimize your marketing campaigns, boost sales conversions, or fine-tune your pricing model, our package provides the insights you need to succeed in today's competitive market.

### Universal Layers in Commercial Algorithms

The models currently implemented in recommendation algorithms:

| #  | Model Name         | model               | Note           |
|----|------------------|---------------------|----------------|
| 1  | Wide and Deep    | WndModel            | Traditional recommendations       |
| 2  | DNN              | DNNModel            | Traditional recommendations       |
| 3  | DeepFM           | DeepFMModel         | Traditional recommendations       |
| 4  | Deep and Cross   | DCNModel            | Traditional recommendations       |
| 5  | XDeepFM          | xDeepFMModel        | Traditional recommendations       |
| 6  | NFM              | NFMModel            | Traditional recommendations       |
| 7  | Tower            | TowerModel          | Traditional recommendations       |
| 8  | FLEN             | FLENModel           | Traditional recommendations       |
| 9  | Fibinet          | FiBiNetModel        | Traditional recommendations       |
| 10 | InterHAt         | InterHAtModel       | Traditional recommendations       |
| 11 | CAN              | CANModel            | Traditional recommendations       |
| 12 | MaskNet          | MaskNetModel        | Traditional recommendations       |
| 13 | ContextNet       | ContextNetModel     | Traditional recommendations       |
| 14 | EDCN             | EDCNModel           | Traditional recommendations       |
| 15 | BertSeq          | Bert4RecModel       | Sequence recommendation       |
| 16 | GRU4Rec          | GRU4RecModel        | Sequence recommendation       |
| 17 | DIN              | DINModel            | Sequence recommendation       |
| 18 | DFN              | DFNModel            | Sequence recommendation       |
| 19 | DCAP             | DCAPModel           | Sequence recommendation       |
| 20 | FBAS             | FBASModel           | Sequence recommendation       |
| 21 | ESMM             | ESMMModel           | Multi objective recommendation     |
| 22 | MMoE             | GeneralMMoEModel    | Multi objective recommendation     |
| 23 | Hard Sharing     | HardSharingModel    | Multi objective recommendation     |
| 24 | Cross Sharing    | CrossSharingModel   | Multi objective recommendation     |
| 25 | Cross Stitch     | CrossStitchModel    | Multi objective recommendation     |
| 26 | PLE              | PLEModel            | Multi objective recommendation     |

### Universal Layers in Commercial Algorithms

In the consolidated algorithms, the following Layer networks have been implemented, which can be conveniently called by higher-level models, or users can directly call the Layer layers to assemble their own models.

| #  | Layer                        | Note                                                                                    |
|----|------------------------------|-----------------------------------------------------------------------------------------|
| 1  | DNNLayer                     | DNN Net                |
| 2  | FMLayer                      | FM Net in DeepFM, NFM  |
| 3  | CrossLayer                   | Cross Net in Deep and Cross |
| 4  | CINLayer                     | CIN Net in XDeepFM          |
| 5  | MultiHeadAttentionLayer      | multi head attention in Bert|
| 6  | SelfAttentionLayer           | scaled dot self attention in Bert|
| 7  | LayerNorm                    | Layer Normalization in Bert|
| 8  | PositionWiseFeedForwardLayer | Position wise feed forward in Bert|
| 9  | TransformerLayer             | Transformer(including multi head attention and LayerNorm) in Bert|
| 10 | TransformerEncoder           | Multi-Transformer in Bert |
| 11 | AutoIntLayer                 | Similar with TransformerLayer|
| 12 | FuseLayer                    | Local Activation Unit in DIN |
| 13 | SENETLayer                   | Squeeze and Excitation Layer |
| 14 | FieldWiseBiInteractionLayer  | FM and MF layer in FLEN|
| 15 | CrossStitchLayer             | Cross-stitch Networks for Multi-task Learning|
| 16 | GeneralMMoELayer             | Modeling Task Relationships in Multi-task Learning with   Multi-gate Mixture-of-Experts |
| 17 | Dice                         | Dice activation function|
| 18 | PositionEncodingLayer        | Positional Encoding Layer in Transformer|
| 19 | CGCGatingNetworkLayer        | task and expert Net in PLE|
| 20 | BiLinearInteractionLayer     | Last feature net in Fibinet|
| 21 | CoActionLayer                | co-action unit layer in CAN|
| 22 | MaskBlockLayer               | MaskBlockLayer in MaskNet|
