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

[A comprehensive toolkit package](https://pypi.org/project/gbiz-torch/) designed to help you accurately predict key metrics such as Click-Through Rates (CTR), Conversion Rates (CVR), uplift, and pricing strategies. Built with state-of-the-art algorithms and user-friendly interfaces, our package streamlines the process of forecasting and decision-making, allowing you to make data-driven choices with confidence. Whether you're looking to optimize your marketing campaigns, boost sales conversions, or fine-tune your pricing model, our package provides the insights you need to succeed in today's competitive market.

## Tutorial
You can learn to use the package by referring to the examples in the directory `./example`

More solution examples will be released soon~

### Useful Eval Matrix
The following eval matrix has been implemented:

| #  | Eval Matrix         | Explanation               | Note           |
|----|------------------|---------------------|----------------|
| 1  | AUC   | Area Under the ROC Curve            | For Classification       |
| 2  | Confusion_Matrix     | Confusion Matrix is a performance measurement for classification            | For Classification       |
| 3  | ACC_F1_score    | Accuracy, Macro-F1 and Weighted-F1            | For Classification       |
| 4  | Top_K_Acc   | top_k_accuracy_score            | For Classification       |
| 5  | Multi_Class_RP    | Multi Class precision, recall and F-beta            | For Classification       |
| 6  | r2_score    | R2_score            | For Classification       |
| 7  | MAE    | Mean Absolute Error            | For Regression       |
| 8  | MSE    | Mean Square Error            | For Regression       |
| 9  | MAPE    | Mean Absolute Percentage Error            | For Regression       |
| 10  | tsne    | t-distributed stochastic neighbor embedding            | For Manifold       |
| 11  | sp_emb    | spectral decomposition to the corresponding graph laplacian            | For Manifold       |

### Universal Layers in Commercial Algorithms

The models currently implemented in recommendation algorithms:

| #  | Model Name         | model               | Note           |
|----|------------------|---------------------|----------------|
| 1  | Wide and Deep    | [WndModel](https://arxiv.org/abs/1606.07792)            | Traditional recommendations       |
| 2  | DNN              | DNNModel            | Traditional recommendations       |
| 3  | DeepFM           | [DeepFMModel](https://arxiv.org/abs/1703.04247)         | Traditional recommendations       |
| 4  | Deep and Cross   | [DCNModel](https://arxiv.org/abs/1708.05123)            | Traditional recommendations       |
| 5  | NFM              | [NFMModel](https://arxiv.org/abs/1708.05027)            | Traditional recommendations       |
| 6  | Tower            | TowerModel          | Traditional recommendations       |
| 7  | FLEN             | [FLENModel](https://arxiv.org/pdf/1911.04690.pdf)           | Traditional recommendations       |
| 8  | Fibinet          | [FiBiNetModel](https://arxiv.org/abs/1905.09433)        | Traditional recommendations       |
| 9 | InterHAt         | [InterHAtModel](https://dl.acm.org/doi/pdf/10.1145/3336191.3371785)       | Traditional recommendations       |
| 10 | CAN              | [CANModel](https://arxiv.org/abs/2011.05625)            | Traditional recommendations       |
| 11 | MaskNet          | [MaskNetModel](https://arxiv.org/abs/2102.07619)        | Traditional recommendations       |
| 12 | ContextNet       | [ContextNetModel](https://arxiv.org/abs/2107.12025)     | Traditional recommendations       |
| 13 | EDCN             | [EDCNModel](https://dl.acm.org/doi/abs/10.1145/3459637.3481915)           | Traditional recommendations       |
| 14 | BertSeq          | [Bert4RecModel](https://arxiv.org/abs/1904.06690)       | Sequence recommendation       |
| 15 | GRU4Rec          | [GRU4RecModel](https://arxiv.org/abs/1511.06939)        | Sequence recommendation       |
| 16 | DIN              | [DINModel](https://arxiv.org/abs/1706.06978)            | Sequence recommendation       |
| 17 | DCAP             | [DCAPModel](https://arxiv.org/abs/2105.08649)           | Sequence recommendation       |
| 18 | FBAS             | FBASModel           | Sequence recommendation       |
| 19 | ESMM             | [ESMMModel](https://arxiv.org/abs/1804.07931)           | Multi objective recommendation     |
| 20 | MMoE             | [GeneralMMoEModel](https://dl.acm.org/doi/10.1145/3219819.3220007)    | Multi objective recommendation     |
| 21 | Hard Sharing     | HardSharingModel    | Multi objective recommendation     |
| 22 | Cross Sharing    | CrossSharingModel   | Multi objective recommendation     |
| 23 | Cross Stitch     | [CrossStitchModel](https://arxiv.org/pdf/1604.03539.pdf)    | Multi objective recommendation     |
| 24 | PLE              | [PLEModel](https://dl.acm.org/doi/10.1145/3383313.3412236)            | Multi objective recommendation     |

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

## Citation
If you find this code useful in your research, please cite it using the following BibTeX:

```bibtex
@software{
  Wang_gbiz_torch_A_comprehensive_2023,
  author = {Wang, Haowen},
  doi = {10.5281/zenodo.10222799},
  month = nov,
  title = {{gbiz_torch: A comprehensive toolkit for predicting key metrics in e-commercial fields}},
  url = {https://github.com/whw199833/gbiz_torch},
  version = {2.0.4},
  year = {2023}
}
```
or following APA:
```APA
Wang, H. (2023). gbiz_torch: A comprehensive toolkit for predicting key metrics in e-commercial fields (Version 2.0.4) [Computer software]. https://doi.org/10.5281/zenodo.10222799
```

## Contact
If you have some questions or some advice, or want to contribute to this repo, do not hesitate to contact me: 

mail: wanghw@zju.edu.cn

wechat: whw199833

