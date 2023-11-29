目前实现的推荐算法中的model：
| #  | 模型名称         | model               | Note           |
|----|------------------|---------------------|----------------|
| 1  | Wide and Deep    | WndModel            | 传统推荐       |
| 2  | DNN              | DNNModel            | 传统推荐       |
| 3  | DeepFM           | DeepFMModel         | 传统推荐       |
| 4  | Deep and Cross   | DCNModel            | 传统推荐       |
| 5  | XDeepFM          | xDeepFMModel        | 传统推荐       |
| 6  | NFM              | NFMModel            | 传统推荐       |
| 7  | Tower            | TowerModel          | 传统推荐       |
| 8  | FLEN             | FLENModel           | 传统推荐       |
| 9  | Fibinet          | FiBiNetModel        | 传统推荐       |
| 10 | GateNet          | GateNetModel        | 传统推荐       |
| 11 | DCNM             | DCNMModel           | 传统推荐       |
| 12 | InterHAt         | InterHAtModel       | 传统推荐       |
| 13 | CAN              | CANModel            | 传统推荐       |
| 14 | MaskNet          | MaskNetModel        | 传统推荐       |
| 15 | ContextNet       | ContextNetModel     | 传统推荐       |
| 16 | EDCN             | EDCNModel           | 传统推荐       |
| 17 | AutoInt          | AutoIntModel        | 序列推荐       |
| 18 | BertSeq          | Bert4RecModel       | 序列推荐       |
| 19 | Multi-CNN        | MultiCNNModel       | 序列推荐       |
| 20 | Caser            | CaserModel          | 序列推荐       |
| 21 | GRU4Rec          | GRU4RecModel        | 序列推荐       |
| 22 | DSIN             | DSINModel           | 序列推荐       |
| 23 | DIN              | DINModel            | 序列推荐       |
| 24 | LSTM DCN         | LSTMDCNModel        | 序列推荐       |
| 25 | DFN              | DFNModel            | 序列推荐       |
| 26 | HGN              | HGNModel            | 序列推荐       |
| 27 | DCAP             | DCAPModel           | 序列推荐       |
| 28 | FBAS             | FBASModel           | 序列推荐       |
| 29 | MIAN             | MIANModel           | 序列推荐       |
| 30 | ESMM             | ESMMModel           | 多目标推荐     |
| 31 | MMoE             | GeneralMMoEModel    | 多目标推荐     |
| 32 | Hard Sharing     | HardSharingModel    | 多目标推荐     |
| 33 | Cross Sharing    | CrossSharingModel   | 多目标推荐     |
| 34 | Cross Stitch     | CrossStitchModel    | 多目标推荐     |
| 35 | Domain MMoE      | DomainMMoEModel     | 多目标推荐     |
| 36 | Bagging MMoE     | BaggingMSLMMoEModel | 多目标推荐     |
| 37 | GMV ESMM         | GMVESMMModel        | 多目标推荐     |
| 38 | PLE              | PLEModel            | 多目标推荐     |
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
| 50 | TIEN             | TIENModel           | 时空推荐       |

推荐算法中的通用Layer
在沉淀的算法中实现了以下的Layer网络，可以方便更上层的model来调用，或者用户可以直接调用Layer层来组装自己的模型。
| #  | Layer                        | Note                                                                                    |
|----|------------------------------|-----------------------------------------------------------------------------------------|
| 1  | DNNLayer                     | 多层dnn网络                                                                             |
| 2  | FMLayer                      | DeepFM, NFM中的FM网络                                                                   |
| 3  | CrossLayer                   | Deep and Cross 中的Cross 网络                                                           |
| 4  | CINLayer                     | XDeepFM中的CIN 网络                                                                     |
| 5  | MultiHeadAttentionLayer      | Bert中的multi head attention 网络                                                       |
| 6  | SelfAttentionLayer           | Bert中的scaled dot self attention 网络                                                  |
| 7  | LayerNorm                    | Bert中的Layer Normalization网络                                                         |
| 8  | PositionWiseFeedForwardLayer | Bert中的Position wise feed forward网络                                                  |
| 9  | TransformerLayer             | Bert中的单层Transformer网络，包含了multi head attention 和 LayerNorm网络                |
| 10 | TransformerEncoder           | Bert中的多层Transformer Layer                                                           |
| 11 | AutoIntLayer                 | 和TransformerLayer 类似                                                                 |
| 12 | FuseLayer                    | DIN模型中的Local Activation Unit网络                                                    |
| 13 | MultiCNNLayer                | MultiCNN中的多通道CNN网络                                                               |
| 14 | SimpleAttnLayer              | MultiCNN中的attention网络                                                               |
| 15 | ResnetLayer                  | Residual Network layer                                                                  |
| 16 | HighWayLayer                 | Highway Networks                                                                        |
| 17 | NextItemAttnLayer            | Next Item Recommendation with Self-Attention                                            |
| 18 | SENETLayer                   | Squeeze and Excitation Layer                                                            |
| 19 | CaserLayer                   | Personalized Top-N Sequential Recommendation via Convolutional   Sequence Embedding     |
| 20 | DSTNLayer                    | Deep Spatio-Temporal Neural Networks for Click-Through Rate   Prediction                |
| 21 | InteresetActLayer            | Deep Session Interest Network for Click through rate   prediction                       |
| 22 | FieldWiseBiInteractionLayer  | FM 和MF layer in FLEN: Leveraging Field for   Scalable CTR Prediction                   |
| 23 | CrossStitchLayer             | Cross-stitch Networks for Multi-task Learning                                           |
| 24 | GeneralMMoELayer             | Modeling Task Relationships in Multi-task Learning with   Multi-gate Mixture-of-Experts |
| 25 | Dice                         | Dice activation function                                                                |
| 26 | Swish                        | Swish activation function                                                               |
| 27 | Gelu                         | GELU activation function                                                                |
| 28 | PositionEncodingLayer        | Positional Encoding Layer in Transformer                                                |
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
| 39 | CGCGatingNetworkLayer        | PLE模型中基础的task和expert网络实现                                                     |
| 40 | BiLinearInteractionLayer     | Fibinet中最后feature的作用网络                                                          |
| 41 | CrossMLayer                  | Cross Mixture模型中的特征交叉                                                           |
| 42 | HierarchicalAttnLayer        | InterHAt中的Hierarchical Attention Aggregation layer                                    |
| 43 | AttnAggregationLayer         | STAN中的Attention Agggregation Layer                                                    |
| 44 | AttnMatchLayer               | STAN中的Attention Match Layer                                                           |
| 45 | CoActionLayer                | CAN中的co-action unit layer                                                             |
| 46 | MaskBlockLayer               | MaskNet中的MaskBlockLayer                                                               |
| 47 | ExternalAttentionLayer       | vision任务中替换self-attention的结构                                                    |
| 48 | gMLPLayer                    | Pay attention to MLP中的序列模型                                                        |
| 49 | AITLayer                     | AITMModel中考虑不同任务之间联系的Layer                                                  |
| 50 | WeightedSeqAggLayer          | 考虑3维序列和候选item之前的求和机制                                                     |
| 51 | PermuteMLPLayer              | vision任务中替换self-attention的MLP机制                                                 |
| 52 | PermutatorBlockLayer         | 类似一层TransformerLayer的采用纯MLP的方案                                               |
| 53 | DropPathLayer                | vision任务中的按照path取dropout的方案      