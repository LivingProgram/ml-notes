# Perceptron
* single unit that takes in a set of input nodes, multiplies those inputs by weights, and adds a bias node, returning an output of 1 or 0
* perceptron visualization: ![perceptron visualization](https://livingprogram.github.io/ml-notes/images/ml-notes_1.jpg)

## Perceptron Algorithm
* For all points (p,q) with label y:
  * Calculate y_hat = step(w_1 * x_1 + w_2 * x_2 + b)
  * If the point is correctly classified, do nothing.
  * If the point is classified positive, but it has a negative label, subtract αp, αq, and α from w_1, w_2, and b respectively.
  * If the point is classified negative, but it has a positive label, add αp, αq, and α to w_1, w_2, b respectively.

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
* softmax equation: ![softmax equation](https://livingprogram.github.io/ml-notes/images/ml-notes_3.jpg)
* softmax with 2 classes: ![softmax 2 classes](https://livingprogram.github.io/ml-notes/images/ml-notes_4.jpg)

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
* Cross Entropy (2 classes): 
* Cross Entropy (m classes): 

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
