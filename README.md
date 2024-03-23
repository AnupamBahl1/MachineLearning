# Repository containing octave code for basic ML Operations.
<br>

## Linear Regression
### Using a single variable to train your model:
<br>
<table>
  <tr>
    <th>Input </th>
    <th>Trained Hypothesis</th>
    <th>Cost Function Contour Plot</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LinearRegression-1.png" width = "300" title="Input">  
    </td>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LinearRegression-2.png" width = "300" title="Hypothesis">  
    </td>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LinearRegression-3.png" width = "300" title="Hypothesis">  
    </td>
  </tr>
</table>

Since cost is a convex function, it's derivative has a singluar local minima. This helps us arrive at the best possible hypothesis with a few iterations.

<br>

### Using multivariate linear regression:
<br>
This method uses 6 different parameters to train the liner regression hypothesis. Plotting cost with the number of iterations shows us that the function coverged. It also confirms that the learning rate selected was accurate (otherwise cost would never converge with increasing number of iterations)

<img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LinearRegression-5.png" width = "300" title="Converging Cost Function">  


<br><br>

## Logistic Regression
<br>

Trying to categorize if a student will be accepted or not at a univeristy given scores from two different examinations. The model I am  trying to derive is a decision boundary between different categories. I am using two categories for demonstration purposes. The same logic can be applied to multiple categories using one category at a time. You can also observe how using input parameters of different degrees results in decision boundaries that could be linear, circular, parabolic etc.

### Using Single Degree Inputs
<table>
  <tr>
    <th>Input </th>
    <th>Trained Hypothesis (Decision Boundary)</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LogisticRegression-1.png" width = "300" title="Input">  
    </td>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LogisticRegression-2.png" width = "300" title="Decision Boundary">  
    </td>
  </tr>
</table>

### Using Inputs with degree 2
<table>
  <tr>
    <th>Input </th>
    <th>Trained Hypothesis (Decision Boundary)</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LogisticRegression-4.png" width = "300" title="Input">  
    </td>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/LogisticRegression-5.png" width = "300" title="Decision Boundary">  
    </td>
  </tr>
</table>

<br><br>

## Neural Networks - Framework
<br>

In the framework project, I am using gradient descent from Logistic regression to calculate the hypothesis, combining it with multiple steps of a neural network to allow multiple modifications of the input parameter. I plan to implement a fully functioning neural network. This code uses 20x20 greyscale images as input. We use pixel intensity values to classify each image into one of 10 classes (1 - 10, number 0 is represented as 10 internally).

<table>
  <tr>
    <th>Input </th>
    <th>Current Input</th>
    <th>Result obtained by using the calcualted hypothesis</th>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/NeuralNetworks-1.png" width = "300" title="Input">  
    </td>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/NeuralNetworks-2.png" width = "300" title="Decision Boundary">  
    </td>
    <td>
      <img src="https://github.com/AnupamBahl1/MachineLearning/blob/main/ML-Outputs/NeuralNetworks-3.png" width = "300" title="Decision Boundary">  
    </td>
  </tr>
</table>


<br><br><br>
