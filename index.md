# Perceptron
* single unit that takes in a set of input nodes, multiplies those inputs by weights, and adds a bias node, returning an output of 1 or 0
* perceptron visualization: ![perceptron visualization](https://livingprogram.github.io/ml-notes/images/ml-notes_1.jpg)

## Perceptron Algorithm
* For all points $$(p,q) \text{ with label } y $$:
  * Calculate $$\hat{y} = step(w_{1} * x_{1} + w_{2} * x_{2} + b)$$
  * If the point is correctly classified: do nothing
  * If the point is classified positive, but it has a negative label: $$w_{1} - \alpha * p$$, $$w_{2} - \alpha * q$$, $$b - \alpha$$
  * If the point is classified negative, but it has a positive label: $$w_{1} + \alpha * p$$, $$w_{2} + \alpha * q$$, $$b + \alpha$$
  * (Where $$\alpha = $$ learning rate)

# Error Function
* measures the model's performance
* must be continuous, not discrete, there should always be a direction towards a more optimal error 

## Sigmoid Activation Function
* sigmoid graph and equation: ![sigmoid function](https://livingprogram.github.io/ml-notes/images/ml-notes_2.jpg)
* takes in all real numbers and outputs a probability between 0 and 1 
* allows a continuous error function

# Softmax Function
* allows you to take linear set of scores for multiple classes and generate probabilities for each class that sum to 0
* exponent function used to avoid negative values (which can cause division by 0 if allowed)
* softmax equation: 
  * Given linear output of scores for $$n$$ classes: 
    $$Z_{1},Z_{2},\ldots,Z_{n}$$
  * $$P(\text{class } i) = \frac{e^{Z_{i}}}{e^{Z_{1}}+e^{Z_{2}}+\ldots+e^{Z_{n}}}$$
* softmax with 2 classes: $$n=2 \\\implies softmax(x)=sigmoid(x)$$
* ![softmax 2 classes](https://livingprogram.github.io/ml-notes/images/ml-notes_4.jpg)

# One-Hot Encoding
* with data that has multiple classes, assign a vector to each class (such that there is a 1 in the row that corresponds to the presence of the class, and the rest are all 0s)

# Maximum Likelihood
* method of picking the model that gives the existing labels the highest probability
* how likely a model's predictions are correct = product of the probabilities that every point is its labeled class
* maximizing the product of probabilities = best model

## Cross-Entropy Error Function
* Motivation for creating cross entropy function:
  * maximum likelihood is able to measure a model's performance
  * products = hard to compute and yield small numbers (instead use sums = easy to calculate)
  * use logs, because of the property log(ab) = log(a) + log(b) (allows our products to turn into sums)
  * log(number_between_0_and_1) = negative numbers (instead use -log() = positive)
  * if we use -log(), minimizing -log() = best model (because before, larger product = better model, and log(large_product) = small number, so now we need to minimize)
* Cross Entropy (2 classes): ![cross entropy 2 classes](https://livingprogram.github.io/ml-notes/images/ml-notes_5.jpg)
* Cross Entropy (2 classes, with W = weights, b = bias): ![cross entropy 2 classes weights bias](https://livingprogram.github.io/ml-notes/images/ml-notes_6.jpg)
* Cross Entropy (n classes): ![cross entropy n classes](https://livingprogram.github.io/ml-notes/images/ml-notes_7.jpg)
* Explanations:
  * y = 1 or 0, therefore only one term in the summation is chosen, and that term will calculate the ln() of the correct probability, then sum the negative ln’s
  * only take -ln() of probabilities that matter, formula when n = 2 turns out to be cross entropy formula for 2 classes

# Gradient Descent

## Gradient Descent Motivation: 
* we want to minimize the cross-entropy error to find the best model
* we need to find the lowest valley in the graph of error function and weights as that is where the error is least
* by taking negative partial derivative of error function with respect to each weight that is the direction to move in towards a lower error and a better model
* therefore we simply need to calculate the gradient of the error

## Gradient Descent Calculation
* Goal = Calculate the gradient of the error = ∇E
* Given m points labeled: ![m points](https://livingprogram.github.io/ml-notes/images/ml-notes_8.jpg)
* Predictions are calculated by the model using: ![prediction formula](https://livingprogram.github.io/ml-notes/images/ml-notes_9.jpg)
* The error for an individual point is: ![individual point error formula](https://livingprogram.github.io/ml-notes/images/ml-notes_10.jpg)
* The overall error is simply the average of individual point errors: ![overall error formula](https://livingprogram.github.io/ml-notes/images/ml-notes_11.jpg)
* The gradient of the error = partial derivatives of error for each weight: ![partial derivatives of error](https://livingprogram.github.io/ml-notes/images/ml-notes_12.jpg)
* First calculate: ![sigmoid derivative proof](https://livingprogram.github.io/ml-notes/images/ml-notes_13.jpg)
* Then calculate: ![prediction partial derivative](https://livingprogram.github.io/ml-notes/images/ml-notes_14.jpg)
* And finally: ![error partial derivative weights](https://livingprogram.github.io/ml-notes/images/ml-notes_15.jpg)
* Similarly: ![error partial derivative bias](https://livingprogram.github.io/ml-notes/images/ml-notes_16.jpg)
* In summary:
  * For a point: ![for a point](https://livingprogram.github.io/ml-notes/images/ml-notes_17.jpg)
  * Conclusion: ![gradient of error formula](https://livingprogram.github.io/ml-notes/images/ml-notes_18.jpg)
* Significance: 
  * gradient = scalar x coordinates of point (scalar = label - prediction)
  * implies: label close to the prediction = small gradient

# Logistic Regression Algorithm
1. Initialize random weights: 
2. For every point:
   * Update weights:
   * Update bias:
3. Repeat until error is small

# Jupyter Cheatsheet
* tab: allows you to complete variable names or list functions of a package within code cell
* shift + tab: lets you see function documentation, variable values 
* (shift + tab)x2: allows you to see more in-depth documentation
* markdown + latex: $ or $$ and can insert latex
* enter: enter edit mode
* escape: enter command mode
* h: show all commands 
* a: create cell above
* b: create cell below
* y: change to code cell
* m: change to markdown cell
* l: toggle line numbers
* d + d: delete cell
* escape + s : saves notebook
* shift + ctrl + p: enter command palette
* arrow keys: move around cells
* x: cut
* c: copy
* v: paste
* space: scroll down
* shift + space: scroll up
* %timeit function(): allows you to time function
* %%timeit: allows you to time entire cell (at top of cell)
* %pdb: turn on interactive debugger (q: turns it off)
* ([additional](http://ipython.readthedocs.io/en/stable/interactive/magics.html) magic commands)
* View > Cell Toolbar > Slideshow: bring up slide cell menu
* jupyter nbconvert notebook.ipynb --to slides: convert to slideshow from file
* jupyter nbconvert notebook.ipynb --to slides --post serve: convert to slideshow and immediately see

# Numpy Cheatsheet
* `x = v[None, :]` ~ add new dimension to array
