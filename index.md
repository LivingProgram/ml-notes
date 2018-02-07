# Perceptron
* single unit that takes in a set of input nodes, multiplies those inputs by weights, and adds a bias node, returning an output of 1 or 0
* perceptron visualization: ![perceptron visualization](https://livingprogram.github.io/ml-notes/images/ml-notes_1.jpg)

## Perceptron Algorithm
* For all points \\((p,q) \text{ with label } y \\):
  * Calculate \\(\hat{y} = step(w_{1} \cdot x_{1} + w_{2} \cdot x_{2} + b)\\)
  * If the point is correctly classified: do nothing
  * If the point is classified positive, but it has a negative label: \\(w_{1} - \alpha \cdot p\\), \\(w_{2} - \alpha \cdot q\\), \\(b - \alpha\\)
  * If the point is classified negative, but it has a positive label: \\(w_{1} + \alpha \cdot p\\), \\(w_{2} + \alpha \cdot q\\), \\(b + \alpha\\)
  * (Where \\(\alpha = \\) learning rate)

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
  * Linear output of scores for \\(n\\) classes: \\(Z_{1},Z_{2},\ldots,Z_{n}\\)
  * $$P(\text{class } i) = \frac{e^{Z_{i}}}{e^{Z_{1}}+e^{Z_{2}}+\ldots+e^{Z_{n}}}$$
* softmax with 2 classes: \\(n=2 \implies softmax(x)=sigmoid(x)\\)

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
  * use logs, because of the property \\(log(ab) = log(a) + log(b)\\) (allows our products to turn into sums)
  * log(number_between_0_and_1) = negative numbers (instead use -log() = positive)
  * if we use -log(), minimizing -log() = best model (because before, larger product = better model, and log(large_product) = small number, so now we need to minimize)
* Notation:
  * \\(m\\) denotes number of training samples
  * \\(X_{i}\\) denotes a specific train sample \\(i\\)
  * \\(n\\) denotes number of features
  * \\(w_{j}\\) denotes a specific feature weight
  * \\(x_{j}\\) denotes a specific input feature
  * \\(\hat{y}_{i}\\) denotes prediction for a specific train sample \\(i\\)
* Calculate predictions:

$$\begin{align}\hat{y}_{i}&=\sigma(WX_{i}+b) \\
\hat{y}_{i}&=\sigma(\sum_{j=1}^{n}w_{j}x_{j}+b) \\
\hat{y}_{i}&=\sigma(w_{1}x_{1}+\ldots+w_{n}x_{n}+b)\end{align}$$

* Cross Entropy (2 classes): 

$$E=-\frac{1}{m}\sum_{i=1}^{m}y_{i}ln(\hat{y}_{i})+(1-y_{i})ln(1-\hat{y}_{i})$$

* Cross Entropy (2 classes, with W = weights, b = bias): 

$$E(W,b)=-\frac{1}{m}\sum_{i=1}^{m}y_{i}ln(\sigma(WX_{i}+b))+(1-y_{i})ln(1-\sigma(WX_{i}+b))$$

* Cross Entropy (n classes): 

$$E=-\sum_{i=1}^{m}\sum_{j=1}^{n}y_{ij}ln(\hat{y}_{ij})$$

* Explanations:
  * y = 1 or 0, therefore only one term in the summation is chosen, and that term will calculate the ln() of the correct probability, then sum the negative ln’s
  * only take -ln() of probabilities that matter, formula when n = 2 turns out to be cross entropy formula for 2 classes

# Gradient Descent

## Gradient Descent Motivation
* we want to minimize the cross-entropy error to find the best model
* we need to find the lowest valley in the graph of error function and weights as that is where the error is least
* by taking negative partial derivative of error function with respect to each weight that is the direction to move in towards a lower error and a better model
* therefore we simply need to calculate the gradient of the error

## Single Train Sample \\(\nabla E_{i}\\)
* Goal = Calculate the gradient of the error for a single training sample, \\(X_{i}\\), \\(= \nabla E_{i}\\)
* Given m training samples labeled: 

$$X_{1},X_{2},\ldots,X_{m}$$

* Individual training sample predictions: 

$$\hat{y}_{i}=\sigma(WX_{i}+b)$$

* Individual training sample error: 

$$E_{i}=-y_{i}ln(\hat{y}_{i})-(1-y_{i})ln(1-\hat{y}_{i})$$

* The gradient of the error = partial derivatives of error for each weight: 

$$\nabla E_{i} = (\frac{\partial}{\partial w_{1}}E_{i},\ldots,\frac{\partial}{\partial w_{n}}E_{i},\frac{\partial}{\partial b}E_{i})$$

* First calculate: 

$$\begin{align}\sigma'(x) &=\frac{d}{dx}\left(\frac{1}{1+e^{-x}}\right) \\
&=\frac{e^{-x}}{(1+e^{-x})^{2}} &&\text{(quotient rule)} \\
&=\frac{1}{1+e^{-x}}\cdot \frac{e^{-x}}{1+e^{-x}} \\
&=\sigma(x)(1-\sigma(x))&&\text{(long division)}\end{align}$$

* Then calculate: 

$$\begin{align}\frac{\partial}{\partial w_{j}}\hat{y}_{i}&=\frac{\partial}{\partial w_{j}}(\sigma(WX_{i}+b)) &&(\hat{y}_{i}\text{ formula)} \\
&= \sigma(WX_{i}+b)(1-\sigma(WX_{i}+b))\cdot\frac{\partial}{\partial w_{j}}(WX_{i}+b) &&(\sigma'(x) \text{ formula)}\\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot\frac{\partial}{\partial w_{j}}(WX_{i}+b) \\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot\frac{\partial}{\partial w_{j}}(w_{1}x_{1}+\ldots+w_{j}x_{j}+\ldots+w_{n}x_{n}+b) \\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot(0+\ldots+x_{j}+\ldots+0) &&\text{(partial derivative)}\\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot x_{j}\end{align}$$

* And finally: 

$$\begin{align}\frac{\partial}{\partial w_{j}}E_{i}&=\frac{\partial}{\partial w_{j}}(-y_{i}ln(\hat{y}_{i})-(1-y_{i})ln(1-\hat{y}_{i})) &&(E_{i}\text{ formula)}\\
&= -y_{i}(\frac{\partial}{\partial w_{j}}(ln(\hat{y}_{i})))-(1-y_{i})(\frac{\partial}{\partial w_{j}}(ln(1-\hat{y}_{i})))\\
&= -y_{i}(\frac{1}{\hat{y}_{i}}\cdot\frac{\partial}{\partial w_{j}}(\hat{y}_{i}))-(1-y_{i})(\frac{1}{1-\hat{y}_{i}}\cdot\frac{\partial}{\partial w_{j}}(1-\hat{y}_{i})) &&\text{(chain rule)}\\
&= -y_{i}(\frac{1}{\hat{y}_{i}}\cdot\hat{y}_{i}(1-\hat{y}_{i})x_{j})-(1-y_{i})(\frac{1}{1-\hat{y}_{i}}\cdot(-1)\hat{y}_{i}(1-\hat{y}_{i})x_{j})&&(\frac{\partial}{\partial w_{j}}\hat{y}_{i}\text{ formula)}\\
&= -y_{i}(1-\hat{y}_{i})x_{j}+(1-y_{i})\hat{y}_{i}\cdot x_{j}\\
&= (-y_{i}+y_{i}\hat{y}_{i}+\hat{y}_{i}-y_{i}\hat{y}_{i})x_{j}\\
&= -(y_{i}-\hat{y}_{i})x_{j}\end{align}$$

* Similarly: 

$$\frac{\partial}{\partial b}E_{i}=-(y_{i}-\hat{y}_{i})$$

* In summary, for a training sample, \\(X_{i}\\), with: 

$$\begin{align}\text{features } &= (x_{1},\ldots,x_{n})\\
\text{label } &= y_{i} \\
\text{prediction } &= \hat{y}_{i}\end{align}$$

$$\begin{align}\nabla E_{i} &= (\frac{\partial}{\partial w_{1}}E_{i},\ldots,\frac{\partial}{\partial w_{n}}E_{i},\frac{\partial}{\partial b}E_{i})\\
&= \left(-(y_{i}-\hat{y}_{i})x_{1},\ldots,-(y_{i}-\hat{y}_{i})x_{n},-(y_{i}-\hat{y}_{i})\right)\\
&= -(y_{i}-\hat{y}_{i})(x_{1},\ldots,x_{n},1)\\
&= (\hat{y}_{i}-y_{i})(x_{1},\ldots,x_{n},1)\end{align}$$

* Significance: 
  * gradient = scalar x coordinates of point (scalar = label - prediction)
  * implies: label close to the prediction = small gradient

## Overall \\(\nabla E\\)
* Goal = Calculate the gradient of the error for over all train samples, \\(= \nabla E\\)
* Given m training samples labeled: 

$$X_{1},X_{2},\ldots,X_{m}$$

* Overall error = average of individual train sample errors: 

$$E=\frac{1}{m}\sum_{i=1}^{m}E_{i}$$

* Overall gradient of error = average of individual train sample gradients:

$$\begin{align}\nabla E &= \frac{1}{m}\sum_{i=1}^{m}\nabla E_{i}\\
&=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_{i}-y_{i})(x_{1},\ldots,x_{n},1)&&(\nabla E_{i}\text{ formula)}\end{align}$$

# Logistic Regression Algorithm

## Batch Size \\(=1\\)
1. Initialize random weights: \\(w_{1},\ldots,w_{n},b\\)
2. For every train sample: \\(X_{1},\ldots,X_{m}\\)
   * Update weights: \\(w_{j}\leftarrow w_{j}-\alpha\frac{\partial}{\partial w_{j}}E_{i}\\)
   * Update bias: \\(b\leftarrow b-\alpha\frac{\partial}{\partial b}E_{i}\\)
3. Repeat until error is small

## Batch Size \\(=m\\)
1. Initialize random weights: \\(w_{1},\ldots,w_{n},b\\)
2. For every batch: 
   * Update weights:\\(w_{j}\leftarrow w_{j}-\alpha\frac{1}{m}\sum_{i=1}^{m}\frac{\partial}{\partial w_{j}}E_{i}\\)
   * Update bias:\\(b\leftarrow b-\alpha\frac{1}{m}\sum_{i=1}^{m}\frac{\partial}{\partial b}E_{i}\\)
3. Repeat until error is small

# Neural Networks
* Built using Multi-Layer Perceptrons: essentially many layers of perceptrons feeding into one another such that each successive perceptron multiplies it's input perceptrons by a learned weight
* Allows us to obtain non-linear models from linear models
* Deep Neural Network: has many layers of neurons
* Multi-Class Classification: apply softmax to the scores of multiple output perceptrons (bounds sum of probabilities for each class between 0 and 1)

## NN Feedforward
* Notation:
  * \\(m\\) denotes number of training samples
  * \\(X_{i}\\) denotes a specific train sample \\(i\\)
  * \\(\hat{y}_{i}\\) denotes prediction for a specific train sample \\(i\\)
  * \\(W^{(k)}_{ij}\\) denotes weight of layer \\(k\\) that connects input neuron \\(i\\) to output neuron \\(j\\)
  * \\(n\\) denotes number of layers in NN
* Calculating NN predictions for train sample \\(X_{i}\\):

$$\hat{y}_{i}=\sigma(W^{(n)}(\sigma(W^{(n-1)}(\ldots(\sigma(W^{(1)}X_{i}))))))$$

* Example \\(n=3\\):

$$\hat{y}_{i}=\sigma(W^{(3)}(\sigma(W^{(2)}(\sigma(W^{(1)}X_{i})))))$$

## Backpropagation
* Method Overview:
  * Perform feedforward
  * Calculate error
  * Propagate error backwards (spread error to all weights)
  * Update all weights using propagated error
  * Loop until satisfied with error
* Intuitive Understanding: 
  * given a model's error, propagate error backwards by decreasing the weights of neurons that had stronger connections over those that had weaker connections
  * the error is caused more by those neurons with strong connections (or large weights), and decreasing their weights will reduce the effects of the erroneous neuron
  * same as single perceptrons, calculate gradient of error function (which is more complex now) and use the gradient to update weights to descend to local minima

### Example NN \\(\nabla E\\) Calculation
* Diagram of "Example NN": ![Example NN](https://livingprogram.github.io/ml-notes/images/ml-notes_19.jpg)
* Notation:
  * \\(x_{j}\\) denotes input feature \\(j\\)
  * \\(W^{(k)}_{ij}\\) denotes weight of layer \\(k\\) that connects input neuron \\(i\\) to output neuron \\(j\\)
  * \\((1)\\) denotes the bias unit
  * \\(h_{j}\\) denotes the \\(j^{\text{th}}\\) neuron in the hidden layer 
  * \\(h\\) denotes the output layer without applying sigmoid function
  * \\(\hat{y}\\) denotes final output prediction
* From the diagram, we have:

$$\begin{align}h_{1}&=W^{(1)}_{11}x_{1}+W^{(1)}_{21}x_{2}+W^{(1)}_{31}\\
h_{2}&=W^{(1)}_{12}x_{1}+W^{(1)}_{22}x_{2}+W^{(1)}_{32}\\
h&=W^{(2)}_{11}\sigma(h_{1})+W^{(2)}_{21}\sigma(h_{2})+W^{(2)}_{31}\\
\hat{y}&=\sigma(h)\\
\hat{y}&=\sigma(W^{(2)}(\sigma(W^{(1)}x)))\end{align}$$

* Final Goal = Calculate the overall gradient of the error \\(= \nabla E\\)
* The gradient of the error = partial derivatives of error with respect to each weight: 

$$\nabla E = \left(\frac{\partial}{\partial W^{(1)}_{11}}E,\ldots,\frac{\partial}{\partial W^{(k)}_{ij}}E,\ldots,\frac{\partial}{\partial W^{(n)}}E\right)$$

* Intermediate Goal = Calculate partial derivative of error with respect to sample weight: \\(\frac{\partial}{\partial W^{(k)}_{ij}}E\\)
* Intermediate Goal = Calculate partial derivative of error with respect to sample weight for single training sample: $$\frac{\partial}{\partial W^{(k)}_{ij}} E_{i}$$
* Overall partial derivative = average of single sample calculation
* Same error function as perceptron (just with a more complex prediction):

$$\begin{align}E(W)&=-\frac{1}{m}\sum_{i=1}^{m}y_{i}ln(\hat{y}_{i})+(1-y_{i})ln(1-\hat{y}_{i})\\
&=E\left(W^{(1)}_{11},W^{(1)}_{12},\ldots,W^{(k)}_{ij},\ldots,W^{(n)}\right)\end{align}$$

* Recall chain rule:

$$\frac{\partial C}{\partial x}=\frac{\partial A}{\partial x}\cdot\frac{\partial B}{\partial A}\cdot\frac{\partial C}{\partial B}$$

* (insert this in detailed section of overview with math included) After calculating gradient, Update weight:

$$W^{(k)}_{ij}\leftarrow W^{(k)}_{ij}-\alpha\frac{\partial}{\partial W^{(k)}_{ij}}E$$

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
