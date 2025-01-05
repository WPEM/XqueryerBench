# Benchmarks 


<table style="width: 100%; border-collapse: collapse; text-align: center;">
  <caption style="font-weight: bold; margin-bottom: 10px;">
    Prediction Accuracy, F1 Score, Precision, and Recall of structure identification, crystal system, and space group classification across different models on simulated and experimental test data.
  </caption>
  <thead>
    <tr style="background-color: #f0f0f0;">
      <th colspan="13" style="font-weight: bold;">Structure Identification</th>
    </tr>
    <tr>
      <th></th>
      <th colspan="4" style="font-weight: bold;">Baselines</th>
      <th colspan="4" style="font-weight: bold;">Simulation</th>
      <th colspan="4" style="font-weight: bold;">RRUFF</th>
    </tr>
    <tr>
      <th>Model</th>
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
      <td>&#10003;</td>
      <td>Ref. 1</td>
      <td>0.005±0.001</td>
      <td>0.001±0.001</td>
      <td>0.001±0.001</td>
      <td>0.005±0.001</td>
      <td>0.003±0.001</td>
      <td>0.001±0.001</td>
      <td>0.001±0.001</td>
      <td>0.002±0.001</td>
    </tr>
    <tr>
      <td>CNN2</td>
      <td>4</td>
      <td>MaxPool</td>
      <td>&times;</td>
      <td>Ref. 2</td>
      <td>0.126±0.099</td>
      <td>0.111±0.088</td>
      <td>0.123±0.097</td>
      <td>0.126±0.099</td>
      <td>0.134±0.034</td>
      <td>0.074±0.020</td>
      <td>0.073±0.019</td>
      <td>0.076±0.020</td>
    </tr>
    <tr>
      <td>CNN3</td>
      <td>3</td>
      <td>None</td>
      <td>&times;</td>
      <td>Ref. 3</td>
      <td>0.180±0.166</td>
      <td>0.162±0.151</td>
      <td>0.171±0.159</td>
      <td>0.180±0.166</td>
      <td>0.155±0.092</td>
      <td>0.088±0.053</td>
      <td>0.088±0.053</td>
      <td>0.090±0.054</td>
    </tr>
    <!-- Repeat for remaining rows -->
  </tbody>
</table>


## Structure Identification
![image](https://github.com/user-attachments/assets/44809248-1201-459a-96a2-6841647d7e8d)

## Crystal System Classification
![image](https://github.com/user-attachments/assets/a36130c1-0256-4597-8734-b9f362200889)

## Space Group Classification
![image](https://github.com/user-attachments/assets/e4adcaaa-47ac-446f-8f96-b66d7c4d31a6)

## Tutorials
- **Training framework**: [model_tutorial](./src/Tutorial.ipynb)

## How to Contribute

1. Upload your score on the branch. If verified, we will add your score to our benchmark (using the simulated test data).
2. Email us your pre-trained model. We will test it on a private test dataset. If there is a large discrepancy in the results, we may request to review your code and training settings.

**No cheating allowed!**
