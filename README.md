# Benchmarks 



## Structure Identification

<table border="1" style="border-collapse: collapse; width: 100%;">
  <caption>
    Prediction Accuracy, F1 Score, Precision, and Recall of structure identification, crystal system, and space group classification across different models on simulated and experimental test data.
  </caption>
  <thead>
    <tr style="background-color: #CCCCCC;">
      <th colspan="13" style="text-align: center;">Structure Identification</th>
    </tr>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4" style="text-align: center;">Baselines</th>
      <th colspan="4" style="text-align: center;">Simulation</th>
      <th colspan="4" style="text-align: center;">RRUFF</th>
    </tr>
    <tr>
      <th>Conv.</th>
      <th>Pooling</th>
      <th>Ensemble</th>
      <th>Ref.</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CNN1</td>
      <td>6</td>
      <td>MaxPool</td>
      <td>&#x2713;</td>
      <td><a href="#ref1">[1]</a></td>
      <td>0.005 ± 0.001</td>
      <td>0.001 ± 0.001</td>
      <td>0.001 ± 0.001</td>
      <td>0.005 ± 0.001</td>
      <td>0.003 ± 0.001</td>
      <td>0.001 ± 0.001</td>
      <td>0.001 ± 0.001</td>
      <td>0.002 ± 0.001</td>
    </tr>
    <tr>
      <td>CNN2</td>
      <td>4</td>
      <td>MaxPool</td>
      <td>&#x2717;</td>
      <td><a href="#ref2">[2]</a></td>
      <td>0.126 ± 0.099</td>
      <td>0.111 ± 0.088</td>
      <td>0.123 ± 0.097</td>
      <td>0.126 ± 0.099</td>
      <td>0.134 ± 0.034</td>
      <td>0.074 ± 0.020</td>
      <td>0.073 ± 0.019</td>
      <td>0.076 ± 0.020</td>
    </tr>
    <tr>
      <td>CNN3</td>
      <td>3</td>
      <td>None</td>
      <td>&#x2717;</td>
      <td><a href="#ref3">[3]</a></td>
      <td>0.180 ± 0.166</td>
      <td>0.162 ± 0.151</td>
      <td>0.171 ± 0.159</td>
      <td>0.180 ± 0.166</td>
      <td>0.155 ± 0.092</td>
      <td>0.088 ± 0.053</td>
      <td>0.088 ± 0.053</td>
      <td>0.090 ± 0.054</td>
    </tr>
    <tr>
      <td>LSTM</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.106 ± 0.060</td>
      <td>0.079 ± 0.045</td>
      <td>0.078 ± 0.045</td>
      <td>0.106 ± 0.060</td>
      <td>0.124 ± 0.036</td>
      <td>0.067 ± 0.021</td>
      <td>0.067 ± 0.021</td>
      <td>0.069 ± 0.021</td>
    </tr>
    <tr>
      <td>GRU</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.115 ± 0.017</td>
      <td>0.082 ± 0.015</td>
      <td>0.080 ± 0.015</td>
      <td>0.115 ± 0.017</td>
      <td>0.084 ± 0.010</td>
      <td>0.044 ± 0.005</td>
      <td>0.044 ± 0.006</td>
      <td>0.046 ± 0.005</td>
    </tr>
  </tbody>
</table>
<tbody>
    <tr>
      <td>Bidirectional-LSTM</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.138 ± 0.065</td>
      <td>0.104 ± 0.050</td>
      <td>0.102 ± 0.050</td>
      <td>0.138 ± 0.065</td>
      <td>0.142 ± 0.025</td>
      <td>0.085 ± 0.013</td>
      <td>0.085 ± 0.013</td>
      <td>0.088 ± 0.014</td>
    </tr>
    <tr>
      <td>Bidirectional-GRU</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.159 ± 0.090</td>
      <td>0.121 ± 0.069</td>
      <td>0.119 ± 0.069</td>
      <td>0.159 ± 0.090</td>
      <td>0.118 ± 0.048</td>
      <td>0.065 ± 0.028</td>
      <td>0.065 ± 0.028</td>
      <td>0.067 ± 0.028</td>
    </tr>
    <tr>
      <td>Transformer</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.253 ± 0.065</td>
      <td>0.227 ± 0.061</td>
      <td>0.222 ± 0.060</td>
      <td>0.253 ± 0.065</td>
      <td>0.173 ± 0.030</td>
      <td>0.117 ± 0.020</td>
      <td>0.115 ± 0.020</td>
      <td>0.120 ± 0.021</td>
    </tr>
    <tr>
      <td>XQueryer</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.362 ± 0.080</td>
      <td>0.320 ± 0.071</td>
      <td>0.312 ± 0.070</td>
      <td>0.362 ± 0.080</td>
      <td>0.225 ± 0.045</td>
      <td>0.155 ± 0.031</td>
      <td>0.153 ± 0.031</td>
      <td>0.158 ± 0.031</td>
    </tr>
  </tbody>
<thead>
    <tr style="background-color: #CCCCCC;">
      <th colspan="13" style="text-align: center;">Crystal System Classification</th>
    </tr>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4" style="text-align: center;">Baselines</th>
      <th colspan="4" style="text-align: center;">Simulation</th>
      <th colspan="4" style="text-align: center;">RRUFF</th>
    </tr>
    <tr>
      <th>Conv.</th>
      <th>Pooling</th>
      <th>Ensemble</th>
      <th>Ref.</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td>CNN1</td>
      <td>6</td>
      <td>MaxPool</td>
      <td>&#x2713;</td>
      <td><a href="#ref1">[1]</a></td>
      <td>0.110 ± 0.005</td>
      <td>0.098 ± 0.005</td>
      <td>0.099 ± 0.005</td>
      <td>0.110 ± 0.005</td>
      <td>0.065 ± 0.004</td>
      <td>0.049 ± 0.003</td>
      <td>0.051 ± 0.004</td>
      <td>0.063 ± 0.004</td>
    </tr>
    <tr>
      <td>CNN2</td>
      <td>4</td>
      <td>MaxPool</td>
      <td>&#x2717;</td>
      <td><a href="#ref2">[2]</a></td>
      <td>0.207 ± 0.022</td>
      <td>0.180 ± 0.020</td>
      <td>0.184 ± 0.020</td>
      <td>0.207 ± 0.022</td>
      <td>0.112 ± 0.014</td>
      <td>0.081 ± 0.010</td>
      <td>0.082 ± 0.010</td>
      <td>0.110 ± 0.014</td>
    </tr>
    <tr>
      <td>LSTM</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.256 ± 0.035</td>
      <td>0.222 ± 0.030</td>
      <td>0.225 ± 0.031</td>
      <td>0.256 ± 0.035</td>
      <td>0.154 ± 0.024</td>
      <td>0.108 ± 0.016</td>
      <td>0.110 ± 0.016</td>
      <td>0.150 ± 0.023</td>
    </tr>
    <tr>
      <td>GRU</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.288 ± 0.043</td>
      <td>0.253 ± 0.037</td>
      <td>0.256 ± 0.037</td>
      <td>0.288 ± 0.043</td>
      <td>0.177 ± 0.032</td>
      <td>0.121 ± 0.022</td>
      <td>0.124 ± 0.023</td>
      <td>0.170 ± 0.031</td>
    </tr>
</tbody>
<thead>
    <tr style="background-color: #CCCCCC;">
      <th colspan="13" style="text-align: center;">Space Group Classification</th>
    </tr>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="4" style="text-align: center;">Baselines</th>
      <th colspan="4" style="text-align: center;">Simulation</th>
      <th colspan="4" style="text-align: center;">RRUFF</th>
    </tr>
    <tr>
      <th>Conv.</th>
      <th>Pooling</th>
      <th>Ensemble</th>
      <th>Ref.</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
</thead>
<tbody>
    <tr>
      <td>CNN1</td>
      <td>6</td>
      <td>MaxPool</td>
      <td>&#x2713;</td>
      <td><a href="#ref1">[1]</a></td>
      <td>0.081 ± 0.003</td>
      <td>0.072 ± 0.003</td>
      <td>0.073 ± 0.003</td>
      <td>0.081 ± 0.003</td>
      <td>0.048 ± 0.003</td>
      <td>0.037 ± 0.002</td>
      <td>0.039 ± 0.003</td>
      <td>0.047 ± 0.003</td>
    </tr>
    <tr>
      <td>CNN2</td>
      <td>4</td>
      <td>MaxPool</td>
      <td>&#x2717;</td>
      <td><a href="#ref2">[2]</a></td>
      <td>0.164 ± 0.018</td>
      <td>0.141 ± 0.015</td>
      <td>0.143 ± 0.016</td>
      <td>0.164 ± 0.018</td>
      <td>0.092 ± 0.012</td>
      <td>0.067 ± 0.009</td>
      <td>0.069 ± 0.009</td>
      <td>0.089 ± 0.011</td>
    </tr>
    <tr>
      <td>LSTM</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.202 ± 0.029</td>
      <td>0.177 ± 0.025</td>
      <td>0.179 ± 0.025</td>
      <td>0.202 ± 0.029</td>
      <td>0.122 ± 0.020</td>
      <td>0.085 ± 0.014</td>
      <td>0.087 ± 0.014</td>
      <td>0.118 ± 0.019</td>
    </tr>
    <tr>
      <td>GRU</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.233 ± 0.035</td>
      <td>0.204 ± 0.030</td>
      <td>0.206 ± 0.030</td>
      <td>0.233 ± 0.035</td>
      <td>0.144 ± 0.026</td>
      <td>0.101 ± 0.018</td>
      <td>0.103 ± 0.019</td>
      <td>0.138 ± 0.025</td>
    </tr>
    <tr>
      <td>XQueryer</td>
      <td colspan="4" style="text-align: center;">-</td>
      <td>0.311 ± 0.042</td>
      <td>0.282 ± 0.038</td>
      <td>0.284 ± 0.038</td>
      <td>0.311 ± 0.042</td>
      <td>0.198 ± 0.033</td>
      <td>0.138 ± 0.023</td>
      <td>0.140 ± 0.023</td>
      <td>0.190 ± 0.031</td>
    </tr>
</tbody>




## Tutorials
- **Training framework**: [model_tutorial](./src/Tutorial.ipynb)

## How to Contribute

1. Upload your score on the branch. If verified, we will add your score to our benchmark (using the simulated test data).
2. Email us your pre-trained model. We will test it on a private test dataset. If there is a large discrepancy in the results, we may request to review your code and training settings.

**No cheating allowed!**
