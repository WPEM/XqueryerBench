# Benchmarks of PXRD Identification

The following table provides the scores achieved on the test dataset we provided for phase identification.


| Model        | # Conv. | # Dropout | # Pooling | # Ensemble | Ref.                                  | Accuracy | F1     | Precision | Recall |
|--------------|---------|-----------|-----------|------------|---------------------------------------|----------|--------|-----------|--------|
| CNN1         | 3       | ✓         | AvgPool   | ✕          | [Park et al., 2017](#park2017classification) | 0        | 0      | 0         | 0      |
| CNN2         | 2       | ✕         | MaxPool   | ✕          | [Lee et al., 2020](#lee2020deep)      | 0        | 0      | 0         | 0      |
| CNN3         | 3       | ✕         | MaxPool   | ✕          | [Lee et al., 2020](#lee2020deep)      | 0        | 0      | 0         | 0      |
| CNN4         | 7       | ✓         | MaxPool   | ✕          | [Wang et al., 2020](#wang2020rapid)   | 0        | 0      | 0         | 0      |
| CNN5         | 3       | ✓         | AvgPool   | ✓          | [Maffettone et al., 2021](#maffettone2021crystallography) | 0        | 0      | 0         | 0      |
| CNN6         | 7       | ✓         | MaxPool   | ✕          | [Dong et al., 2021](#dong2021deep)    | 0        | 0      | 0         | 0      |
| CNN7         | 6       | ✓         | MaxPool   | ✓          | [Szymanski et al., 2021](#szymanski2021probabilistic) | 0.006    | 0      | 0         | 0.006  |
| CNN8         | 14      | ✓         | MaxPool   | ✕          | [Lee et al., 2022](#lee2022powder)    | 0        | 0      | 0         | 0      |
| CNN9         | 3       | ✓         | MaxPool   | ✕          | [Le et al., 2023](#le2023deep)        | 0        | 0      | 0         | 0      |
| CNN10        | 4       | ✓         | MaxPool   | ✕          | [Le et al., 2023](#le2023deep)        | 0.170    | 0.147  | 0.165     | 0.170  |
| CNN11        | 3       | ✓         | None      | ✕          | [Salgado et al., 2023](#salgado2023automated) | 0.304    | 0.274  | 0.288     | 0.304  |
| MLP          | -       | -         | -         | -          | -                                     | 0        | 0      | 0         | 0      |
| RNN          | -       | -         | -         | -          | -                                     | 0        | 0      | 0         | 0      |
| LSTM         | -       | -         | -         | -          | -                                     | 0.135    | 0.104  | 0.103     | 0.135  |
| GRU          | -       | -         | -         | -          | -                                     | 0.113    | 0.079  | 0.077     | 0.113  |
| Bi-RNN       | -       | -         | -         | -          | -                                     | 0        | 0      | 0         | 0      |
| Bi-LSTM      | -       | -         | -         | -          | -                                     | 0.343    | 0.309  | 0.322     | 0.343  |
| Bi-GRU       | -       | -         | -         | -          | -                                     | 0.398    | 0.362  | 0.377     | 0.398  |
| Transformer  | -       | -         | -         | -          | -                                     | 0        | 0      | 0         | 0      |
| SegRNN       | -       | -         | -         | -          | -                                     | 0.278    | 0.244  | 0.268     | 0.278  |
| iTransformer | -       | -         | -         | -          | -                                     | 0.318    | 0.291  | 0.306     | 0.318  |
| PatchTST     | -       | -         | -         | -          | -                                     | 0.187    | 0.163  | 0.180     | 0.187  |
| JADE Pro 8.9 | -       | -         | -         | -          | -                                     | 0.202    | 0.196  | 0.181     | 0.201  |
| Xqueryer     | -       | -         | -         | -          | -                                     | **0.725**| **0.717**| **0.734**| **0.711** |

<a name="park2017classification">Classification of crystal structure using a convolutional neural network.</a>

<a name="lee2020deep">Lee et al., 2020. A deep-learning technique for phase identification in multiphase inorganic compounds using synthetic XRD powder patterns.</a>

<a name="wang2020rapid">Wang et al., 2020. Rapid identification of X-ray diffraction patterns based on very limited data by interpretable convolutional neural networks.</a>

<a name="maffettone2021crystallography">Maffettone et al., 2021. Crystallography companion agent for high-throughput materials discovery.</a>

<a name="dong2021deep">Dong et al., 2021. A deep convolutional neural network for real-time full profile analysis of big powder diffraction data.</a>

<a name="szymanski2021probabilistic">Szymanski et al., 2021. Probabilistic deep learning approach to automate the interpretation of multi-phase diffraction spectra.</a>

<a name="lee2022powder">Lee et al., 2022. Powder X-ray diffraction pattern is all you need for machine-learning-based symmetry identification and property prediction</a>


<a name="le2023deep">Le et al., 2023. Deep Learning Models to Identify Common Phases across Material Systems from X-ray Diffraction.</a>

<a name="salgado2023automated">Salgado et al., 2023. Automated classification of big X-ray diffraction data using deep learning models.</a>

## How to Contribute

1. Upload your score on the branch. If verified, we will add your score to our benchmark (using the simulated test data).
2. Email us your pre-trained model. We will test it on a private test dataset. If there is a large discrepancy in the results, we may request to review your code and training settings.

**No cheating allowed!**
