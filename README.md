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
<!-- 
| 10 | GateNet          | GateNetModel        | 传统推荐       |
| 11 | DCNM             | DCNMModel           | 传统推荐       |
| 17 | AutoInt          | AutoIntModel        | 序列推荐       |
| 19 | Multi-CNN        | MultiCNNModel       | 序列推荐       |
| 20 | Caser            | CaserModel          | 序列推荐       |
| 22 | DSIN             | DSINModel           | 序列推荐       |
| 24 | LSTM DCN         | LSTMDCNModel        | 序列推荐       |
| 26 | HGN              | HGNModel            | 序列推荐       |
| 29 | MIAN             | MIANModel           | 序列推荐       |
| 35 | Domain MMoE      | DomainMMoEModel     | 多目标推荐     |
| 36 | Bagging MMoE     | BaggingMSLMMoEModel | 多目标推荐     |
| 37 | GMV ESMM         | GMVESMMModel        | 多目标推荐     |
| 39 | AITM             | AITMModel           | 多目标推荐     |
| 40 | MoSE             | MoSEModel           | 多目标序列推荐 |
| 41 | 深度单调定价网络 | DIPNModel           | 定价算法       |
| 42 | 单调MF           | MonoMFModel         | 定价算法       |
| 43 | Cox回归          | CoxRegressionModel  | 定价算法       |
| 44 | DLCM             | DLCMModel           | 重排算法       |
| 45 | PRM              | PRMModel            | 重排算法       |
| 46 | SRGA             | SRGAModel           | 重排算法       |
| 47 | XPA              | XPAModel            | 重排算法       |
| 48 | HGME             | HGMEModel           | 多目标重排算法 |
| 49 | STAN             | STANModel           | 时空推荐       |
| 50 | TIEN             | TIENModel           | 时空推荐       | -->

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


<!-- | 13 | MultiCNNLayer                | MultiCNN中的多通道CNN网络                                                               |
| 14 | SimpleAttnLayer              | MultiCNN中的attention网络                                                               |
| 15 | ResnetLayer                  | Residual Network layer                                                                  |
| 16 | HighWayLayer                 | Highway Networks                                                                        |
| 17 | NextItemAttnLayer            | Next Item Recommendation with Self-Attention                                            |
| 19 | CaserLayer                   | Personalized Top-N Sequential Recommendation via Convolutional   Sequence Embedding     |
| 20 | DSTNLayer                    | Deep Spatio-Temporal Neural Networks for Click-Through Rate   Prediction                |
| 21 | InteresetActLayer            | Deep Session Interest Network for Click through rate   prediction                       |
| 26 | Swish                        | Swish activation function                                                               |
| 27 | Gelu                         | GELU activation function                                                                |
| 29 | IsotonicLayer                | 单调性层，关于输入金额单调                                                              |
| 30 | MonoMFLayer                  | 单调MF层，会对输入金额计算单调的embedding                                               |
| 31 | MonotonicEmbeddingLayer      | 输入金额特征，输出单调embedding                                                         |
| 32 | CoxRegressionLayer           | Cox回归计算层                                                                           |
| 33 | NegativeSampler              | 负采样计算                                                                              |
| 34 | SynthesizerLayer             | self-attention的高效实现的variants                                                      |
| 35 | LocalGlobalLayer             | self-attention的中增加local module，用在ELVM模型中                                      |
| 36 | LinFormerAttnLayer           | self-attention的高效实现的variants                                                      |
| 37 | GatedTanhLayer               | GateNet中的gate实现                                                                     |
| 38 | GatedDNNLayer                | GateNet中的gate+DNN实现                                                                 |
| 41 | CrossMLayer                  | Cross Mixture模型中的特征交叉                                                           |
| 42 | HierarchicalAttnLayer        | InterHAt中的Hierarchical Attention Aggregation layer                                    |
| 43 | AttnAggregationLayer         | STAN中的Attention Agggregation Layer                                                    |
| 44 | AttnMatchLayer               | STAN中的Attention Match Layer                                                           |
| 47 | ExternalAttentionLayer       | vision任务中替换self-attention的结构                                                    |
| 48 | gMLPLayer                    | Pay attention to MLP中的序列模型                                                        |
| 49 | AITLayer                     | AITMModel中考虑不同任务之间联系的Layer                                                  |
| 50 | WeightedSeqAggLayer          | 考虑3维序列和候选item之前的求和机制                                                     |
| 51 | PermuteMLPLayer              | vision任务中替换self-attention的MLP机制                                                 |
| 52 | PermutatorBlockLayer         | 类似一层TransformerLayer的采用纯MLP的方案                                               |
| 53 | DropPathLayer                | vision任务中的按照path取dropout的方案       -->