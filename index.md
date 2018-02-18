
# Perceptron
- single unit that takes in a set of input nodes, multiplies those inputs by weights, and adds a bias node, returning an output of 1 or 0
- perceptron visualization: ![perceptron visualization](https://livingprogram.github.io/ml-notes/images/ml-notes_1.jpg)

## Perceptron Algorithm
- For all points $$(p,q) \text{ with label } y $$:
  - Calculate $$\hat{y} = step(w_{1} \cdot x_{1} + w_{2} \cdot x_{2} + b)$$
  - If the point is correctly classified: do nothing
  - If the point is classified positive, but it has a negative label: $$w_{1} - \alpha \cdot p$$, $$w_{2} - \alpha \cdot q$$, $$b - \alpha$$
  - If the point is classified negative, but it has a positive label: $$w_{1} + \alpha \cdot p$$, $$w_{2} + \alpha \cdot q$$, $$b + \alpha$$
  - (Where $$\alpha = $$ learning rate)

# Error Function
- measures the model's performance
- must be continuous, not discrete, there should always be a direction towards a more optimal error

# Sigmoid Activation Function
- sigmoid graph and equation: ![sigmoid function](https://livingprogram.github.io/ml-notes/images/ml-notes_2.jpg)
- takes in all real numbers and outputs a probability between 0 and 1
- allows a continuous error function

# Softmax Function
- allows you to take linear set of scores for multiple classes and generate probabilities for each class that sum to 0
- exponent function used to avoid negative values (which can cause division by 0 if allowed)
- softmax equation:
  - Linear output of scores for $$n$$ classes: $$Z_{1},Z_{2},\ldots,Z_{n}$$
  - $$P(\text{class } i) = \frac{e^{Z_{i}}}{e^{Z_{1}}+e^{Z_{2}}+\ldots+e^{Z_{n}}}$$
- softmax with 2 classes: $$n=2 \implies softmax(x)=sigmoid(x)$$

# One-Hot Encoding
- with data that has multiple classes, assign a vector to each class (such that there is a 1 in the row that corresponds to the presence of the class, and the rest are all 0s)

# Maximum Likelihood
- method of picking the model that gives the existing labels the highest probability
- how likely a model's predictions are correct = product of the probabilities that every point is its labeled class
- maximizing the product of probabilities = best model

# Cross-Entropy Error Function
## Cross-Entropy Motivation
- maximum likelihood is able to measure a model's performance
- products = hard to compute and yield small numbers (instead use sums = easy to calculate)
- use logs, because of the property $$log(ab) = log(a) + log(b)$$ (allows our products to turn into sums)
- log(number_between_0_and_1) = negative numbers (instead use -log() = positive)
- if we use -log(), minimizing -log() = best model (because before, larger product = better model, and log(large_product) = small number, so now we need to minimize)

## Cross-Entropy Notation
- $$m$$ denotes number of training samples
- $$X_{i}$$ denotes a specific train sample $$i$$
- $$n$$ denotes number of features
- $$w_{j}$$ denotes a specific feature weight
- $$x_{j}$$ denotes a specific input feature
- $$\hat{y}_{i}$$ denotes prediction for a specific train sample $$i$$

## Cross-Entropy Equations
- Calculate predictions:

$$\begin{align}\hat{y}_{i}&=\sigma(WX_{i}+b) \\
\hat{y}_{i}&=\sigma(\sum_{j=1}^{n}w_{j}x_{j}+b) \\
\hat{y}_{i}&=\sigma(w_{1}x_{1}+\ldots+w_{n}x_{n}+b)\end{align}$$

- Cross Entropy (2 classes):

$$E=-\frac{1}{m}\sum_{i=1}^{m}y_{i}ln(\hat{y}_{i})+(1-y_{i})ln(1-\hat{y}_{i})$$

- Cross Entropy (2 classes, with W = weights, b = bias):

$$E(W,b)=-\frac{1}{m}\sum_{i=1}^{m}y_{i}ln(\sigma(WX_{i}+b))+(1-y_{i})ln(1-\sigma(WX_{i}+b))$$

- Cross Entropy (n classes):

$$E=-\sum_{i=1}^{m}\sum_{j=1}^{n}y_{ij}ln(\hat{y}_{ij})$$

## Cross-Entropy Explanations
- y = 1 or 0, therefore only one term in the summation is chosen, and that term will calculate the ln() of the correct probability, then sum the negative ln’s
- only take -ln() of probabilities that matter, formula when n = 2 turns out to be cross entropy formula for 2 classes

# Gradient Descent

## Gradient Descent Motivation
- we want to minimize the cross-entropy error to find the best model
- we need to find the lowest valley in the graph of error function and weights as that is where the error is least
- by taking negative partial derivative of error function with respect to each weight that is the direction to move in towards a lower error and a better model
- therefore we simply need to calculate the gradient of the error
- also gradient = scalar x coordinates of point (scalar = label - prediction), this means label close to the prediction = small gradient

## Gradient Descent Notation
- $$(x_{1},\ldots,x_{n})$$ : features for a specific training sample
- $$X$$ : vector of features
- $$X_{1},X_{2},\ldots,X_{m}$$ : features vectors for $$m$$ training samples
- $$y$$ : label for training samples
- $$\hat{y}$$ : predictions from algorithm for training samples
- $$E$$ : cross entropy error
- $$\nabla E$$ : gradient of error
- $$(w_{1},\ldots,w_{n})$$ : weights of algorithm
- $$\sigma(x)$$ : sigmoid function
- $$\text{subscript}\ \ i$$ : for specific training sample

## GD Single Train Sample $$\nabla E_{i}$$
For a single training sample, $$X_{i}$$:

$$\begin{align}\nabla E_{i} &= (\frac{\partial}{\partial w_{1}}E_{i},\ldots,\frac{\partial}{\partial w_{n}}E_{i},\frac{\partial}{\partial b}E_{i})\\
&= (\hat{y}_{i}-y_{i})(x_{1},\ldots,x_{n},1)\end{align}$$

### Proof.

$$
\text{Individual training sample predictions:}\\
\hat{y}_{i}=\sigma(WX_{i}+b)
$$

$$
\text{Individual training sample error:}\\
E_{i}=-y_{i}ln(\hat{y}_{i})-(1-y_{i})ln(1-\hat{y}_{i})
$$

$$
\text{Gradient is equal to partial derivatives of error for each weight:}\\
\nabla E_{i} = (\frac{\partial}{\partial w_{1}}E_{i},\ldots,\frac{\partial}{\partial w_{n}}E_{i},\frac{\partial}{\partial b}E_{i})\\\\
$$

$$\text{Sigmoid function derivative:}\\
\begin{align}
\sigma'(x) &=\frac{d}{dx}\left(\frac{1}{1+e^{-x}}\right) \\
&=\frac{e^{-x}}{(1+e^{-x})^{2}} &&\text{(quotient rule)} \\
&=\frac{1}{1+e^{-x}}\cdot \frac{e^{-x}}{1+e^{-x}} \\
&=\sigma(x)(1-\sigma(x))&&\text{(long division)}\\\\
\end{align}$$

$$\text{Partial derivative of prediction:}$$

$$
\begin{align}
\frac{\partial}{\partial w_{j}}\hat{y}_{i}&=\frac{\partial}{\partial w_{j}}(\sigma(WX_{i}+b)) &&(\hat{y}_{i}\text{ formula)} \\
&= \sigma(WX_{i}+b)(1-\sigma(WX_{i}+b))\cdot\frac{\partial}{\partial w_{j}}(WX_{i}+b) &&(\sigma'(x) \text{ formula)}\\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot\frac{\partial}{\partial w_{j}}(WX_{i}+b) \\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot\frac{\partial}{\partial w_{j}}(w_{1}x_{1}+\ldots+w_{j}x_{j}+\ldots+w_{n}x_{n}+b) \\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot(0+\ldots+x_{j}+\ldots+0) &&\text{(partial derivative)}\\
&= \hat{y}_{i}(1-\hat{y}_{i})\cdot x_{j}\\\\
\end{align}$$

$$\text{Partial derivative of error:}$$

$$
\begin{align}\frac{\partial}{\partial w_{j}}E_{i}&=\frac{\partial}{\partial w_{j}}(-y_{i}ln(\hat{y}_{i})-(1-y_{i})ln(1-\hat{y}_{i})) &&(E_{i}\text{ formula)}\\
&= -y_{i}(\frac{\partial}{\partial w_{j}}(ln(\hat{y}_{i})))-(1-y_{i})(\frac{\partial}{\partial w_{j}}(ln(1-\hat{y}_{i})))\\
&= -y_{i}(\frac{1}{\hat{y}_{i}}\cdot\frac{\partial}{\partial w_{j}}(\hat{y}_{i}))-(1-y_{i})(\frac{1}{1-\hat{y}_{i}}\cdot\frac{\partial}{\partial w_{j}}(1-\hat{y}_{i})) &&\text{(chain rule)}\\
&= -y_{i}(\frac{1}{\hat{y}_{i}}\cdot\hat{y}_{i}(1-\hat{y}_{i})x_{j})-(1-y_{i})(\frac{1}{1-\hat{y}_{i}}\cdot(-1)\hat{y}_{i}(1-\hat{y}_{i})x_{j})&&(\frac{\partial}{\partial w_{j}}\hat{y}_{i}\text{ formula)}\\
&= -y_{i}(1-\hat{y}_{i})x_{j}+(1-y_{i})\hat{y}_{i}\cdot x_{j}\\
&= (-y_{i}+y_{i}\hat{y}_{i}+\hat{y}_{i}-y_{i}\hat{y}_{i})x_{j}\\
&= -(y_{i}-\hat{y}_{i})x_{j}\\\\
\end{align}$$

$$\text{By a similar proof:}$$

$$\frac{\partial}{\partial b}E_{i}=-(y_{i}-\hat{y}_{i})\\\\$$

$$\text{Gradient of error:}$$

$$\begin{align}
\nabla E_{i} &= (\frac{\partial}{\partial w_{1}}E_{i},\ldots,\frac{\partial}{\partial w_{n}}E_{i},\frac{\partial}{\partial b}E_{i})\\
&= \left(-(y_{i}-\hat{y}_{i})x_{1},\ldots,-(y_{i}-\hat{y}_{i})x_{n},-(y_{i}-\hat{y}_{i})\right)\\
&= -(y_{i}-\hat{y}_{i})(x_{1},\ldots,x_{n},1)\\
&= (\hat{y}_{i}-y_{i})(x_{1},\ldots,x_{n},1)\ \ \ \ \blacksquare\\\\
\end{align}$$

## GD Overall $$\nabla E$$
For $$m$$ training samples:

$$\nabla E = \frac{1}{m}\sum_{i=1}^{m}(\hat{y}_{i}-y_{i})(x_{1},\ldots,x_{n},1)$$

### Proof.

$$\text{Overall error = average of individual train sample errors:}$$

$$E=\frac{1}{m}\sum_{i=1}^{m}E_{i}$$

$$\text{Overall gradient of error = average of individual train sample gradients:}$$

$$\begin{align}\nabla E &= \frac{1}{m}\sum_{i=1}^{m}\nabla E_{i}\\
&=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_{i}-y_{i})(x_{1},\ldots,x_{n},1)\ \ \ \ \blacksquare\\\\\end{align}$$

# Logistic Regression Algorithm

## When Batch Size $$=1$$
1. Initialize random weights: $$w_{1},\ldots,w_{n},b$$
2. For every train sample: $$X_{1},\ldots,X_{m}$$
   - Update weights: $$w_{j}\leftarrow w_{j}-\alpha\frac{\partial}{\partial w_{j}}E_{i}$$
   - Update bias: $$b\leftarrow b-\alpha\frac{\partial}{\partial b}E_{i}$$
3. Repeat until error is small

## When Batch Size $$=m$$
1. Initialize random weights: $$w_{1},\ldots,w_{n},b$$
2. For every batch:
   - Update weights:$$w_{j}\leftarrow w_{j}-\alpha\frac{1}{m}\sum_{i=1}^{m}\frac{\partial}{\partial w_{j}}E_{i}$$
   - Update bias:$$b\leftarrow b-\alpha\frac{1}{m}\sum_{i=1}^{m}\frac{\partial}{\partial b}E_{i}$$
3. Repeat until error is small

# Neural Networks
- Built using Multi-Layer Perceptrons: essentially many layers of perceptrons feeding into one another such that each successive perceptron multiplies it's input perceptrons by a learned weight
- Allows us to obtain non-linear models from linear models
- Deep Neural Network: has many layers of neurons
- Multi-Class Classification: apply softmax to the scores of multiple output perceptrons (bounds sum of probabilities for each class between 0 and 1)

## NN Notation
- $$m$$ : number of training samples
- $$(x_{1},\ldots,x_{n})$$ : features for a specific training sample
- $$X$$ : vector of features
- $$X_{1},X_{2},\ldots,X_{m}$$ : features vectors for $$m$$ training samples
- $$y$$ : label for training samples
- $$\hat{y}$$ : predictions from algorithm for training samples
- $$E$$ : cross entropy error
- $$\nabla E$$ : gradient of error
- $$\sigma(x)$$ : sigmoid function
- $$W^{(l)}_{ij}$$ : weight of layer $$l$$ that connects input neuron $$i$$ to output neuron $$j$$
- $$W^{(l)}$$ : weights vector for layer $$l$$
- $$L$$ : number of layers, including input layer
- $$z^{(l)}_{j}$$ denotes the output of the $$j^{\text{th}}$$ neuron in the $$l^{\text{th}}$$ layer before applying sigmoid function
- $$a^{(l)}_{j}$$ denotes the output of the $$j^{\text{th}}$$ neuron in the $$l^{\text{th}}$$ layer after applying sigmoid function
- $$\text{subscript}\ \ i$$ : for specific training sample

## NN Feedforward Equations

- Calculating NN predictions for train sample $$X_{i}$$:

$$\hat{y}_{i}=\sigma(W^{(L-1)}(\sigma(W^{(L-2)}(\ldots(\sigma(W^{(1)}X_{i}))))))$$

- Example when $$L=4$$:

$$\hat{y}_{i}=\sigma(W^{(3)}(\sigma(W^{(2)}(\sigma(W^{(1)}X_{i})))))$$

## NN Backpropagation
### Backprop Method Overview
- Perform feedforward
- Calculate error
- Propagate error backwards (spread error to all weights)
- Update all weights using propagated error
- Loop until satisfied with error

### Backprop Intuitive Understanding
- given a model's error, propagate error backwards by decreasing the weights of neurons that had stronger connections over those that had weaker connections
- the error is caused more by those neurons with strong connections (or large weights), and decreasing their weights will reduce the effects of the erroneous neuron
- same as single perceptrons, calculate gradient of error function (which is more complex now) and use the gradient to update weights to descend to local minima

### Sample NN Diagram
![Example NN](https://livingprogram.github.io/ml-notes/images/ml-notes_21.jpg)

### Calculating $$\frac{\partial}{\partial W^{(1)}_{11}}E$$
For the sample NN:

$$\frac{\partial}{\partial W^{(1)}_{11}}E=\ldots$$

#### Proof.

$$\text{For simplicity, Let:}$$

$$\begin{align}
x_1=a_1^{(1)},\ x_2&=a_2^{(1)},\ x_3=a_3^{(1)}, \\
\hat{y}&=a_1^{(4)}
\end{align}$$

$$\text{Equations from sample NN:}$$

$$\begin{align}
z_1^{(2)} &= W_{11}^{(1)}a_1^{(1)} + W_{21}^{(1)}a_2^{(1)} + W_{31}^{(1)}a_3^{(1)} \\
a_1^{(2)} &= \sigma(z_1^{(2)}) \\
z_1^{(3)} &= W_{11}^{(2)}a_1^{(2)} + W_{21}^{(2)}a_2^{(2)} + W_{31}^{(2)}a_3^{(2)} + W_{41}^{(2)}a_4^{(2)}\\
a_1^{(3)} &= \sigma(z_1^{(3)}) \\
z_2^{(3)} &= W_{12}^{(2)}a_1^{(2)} + W_{22}^{(2)}a_2^{(2)} + W_{32}^{(2)}a_3^{(2)} + W_{42}^{(2)}a_4^{(2)}\\
a_2^{(3)} &= \sigma(z_2^{(3)}) \\
z_3^{(3)} &= W_{13}^{(2)}a_1^{(2)} + W_{23}^{(2)}a_2^{(2)} + W_{33}^{(2)}a_3^{(2)} + W_{43}^{(2)}a_4^{(2)}\\
a_3^{(3)} &= \sigma(z_3^{(3)}) \\
z_1^{(4)} &= W_{11}^{(3)}a_1^{(3)} + W_{21}^{(3)}a_2^{(3)} + W_{31}^{(3)}a_3^{(3)} \\
a_1^{(4)} &= \sigma(z_1^{(4)}) \\
E &= -y\ln(a_1^{(4)})-(1-y)\ln(1-a_1^{(4)})\\\\
\end{align}$$

$$\text{Recall chain rule:}$$

$$\frac{\partial C}{\partial x} = \frac{\partial A}{\partial x} \cdot \frac{\partial B}{\partial A} \cdot \frac{\partial C}{\partial B}\\\\$$

$$\text{The following is incorrect use of chain rule:}$$

$$
\frac{\partial E}{\partial W_{11}^{(1)}} = \frac{\partial E}{\partial a_1^{(4)}} \cdot \frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} \cdot \frac{\partial z_1^{(4)}}{\partial a_1^{(3)}} \cdot \frac{\partial a_1^{(3)}}{\partial z_1^{(3)}} \cdot \frac{\partial z_1^{(3)}}{\partial a_1^{(2)}} \cdot \frac{\partial a_1^{(2)}}{\partial z_1^{(2)}} \cdot \frac{\partial z_1{(2)}}{\partial W_{11}^{(1)}} \\\\
$$

$$\begin{align}
\text{Incorrect because it only propagates error from }& a_1^{(4)}\to a_1^{(3)}\to a_1^{(2)}\to W_{11}^{(1)} \\
\text{And neglects error that also propagates from }& a_1^{(4)}\to a_2^{(3)}\to a_1^{(2)}\to W_{11}^{(1)} \\
\text{as well as from }& a_1^{(4)}\to a_3^{(3)}\to a_1^{(2)}\to W_{11}^{(1)} \\
\end{align}$$

(Work in Progress below... )

- Final Goal = Calculate the overall gradient of the error:

$$\nabla E$$

- The gradient of the error = partial derivatives of error with respect to each weight:

$$\nabla E = \left(\frac{\partial}{\partial W^{(1)}_{11}}E,\ldots,\frac{\partial}{\partial W^{(k)}_{ij}}E,\ldots,\frac{\partial}{\partial W^{(3)}_{31}}E\right)$$

- Intermediate Goal = Calculate partial derivative of error with respect to sample weight:

$$\frac{\partial}{\partial W^{(1)}_{11}}E$$

- Intermediate Goal = Calculate partial derivative of error with respect to sample weight for single training sample:

$$\frac{\partial}{\partial W^{(1)}_{11}} E_{i}$$

- (All subscript i's will be removed to simplify calculation of partial derivatives)
- From the diagram, we have:

$$\begin{align}z_{1}^{(2)}&=W^{(1)}X=W_{11}^{(1)}x_{1}+W_{21}^{(1)}x_{2}+W_{31}^{(1)}\\
a_{1}^{(2)}&=\sigma(z_{1}^{(2)})\\
z_{1}^{(3)}&=W^{(2)}a^{(2)}=W_{11}^{(2)}a_{1}^{(2)}+W_{21}^{(2)}a_{2}^{(2)}+W_{31}^{(2)}a_{3}^{(2)}+W_{41}^{(2)}\\
a_{1}^{(3)}&=\sigma(z_{1}^{(3)})\\
z_{1}^{(4)}&=W^{(3)}a^{(3)}=W_{11}^{(3)}a_{1}^{(3)}+W_{21}^{(3)}a_{2}^{(3)}+W_{31}^{(3)}\\
\hat{y}&=\sigma(z_{1}^{(4)})\\
\hat{y}&=\sigma(W^{(3)}(\sigma(W^{(2)}(\sigma(W^{(1)}X)))))\end{align}$$

- Same error function as perceptron (just with a more complex prediction):

$$E=-yln(\hat{y})-(1-y)ln(1-\hat{y})$$

- Recall chain rule:

$$\frac{\partial C}{\partial x}=\frac{\partial A}{\partial x}\cdot\frac{\partial B}{\partial A}\cdot\frac{\partial C}{\partial B}$$

- From the chain rule:

$$\frac{\partial}{\partial W^{(1)}_{11}}E=\frac{\partial E}{\partial \hat{y}}\cdot\frac{\partial \hat{y}}{\partial z_{1}^{(4)}}\cdot\frac{\partial z_{1}^{(4)}}{\partial a_{1}^{(3)}}\cdot\frac{\partial a_{1}^{(3)}}{\partial z_{1}^{(3)}}\cdot\frac{\partial z_{1}^{(3)}}{\partial a_{1}^{(2)}}\cdot\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}\cdot\frac{\partial z_{1}^{(2)}}{\partial W^{(1)}_{11}}$$

- Calculating partial derivatives:

$$\begin{align}\frac{\partial E}{\partial \hat{y}}&=\frac{\partial}{\partial \hat{y}}(-yln(\hat{y})-(1-y)ln(1-\hat{y}))\\
&=-y\cdot\frac{\partial}{\partial \hat{y}}(ln(\hat{y})-(1-y)\cdot\frac{\partial}{\partial \hat{y}}(ln(1-\hat{y}))\\
&=-y\cdot\frac{1}{\hat{y}}\cdot 1-(1-y)\cdot\frac{1}{1- \hat{y}}\cdot -1\\
&=\frac{\hat{y}-y}{\hat{y}(1-\hat{y})}\\
\frac{\partial \hat{y}}{\partial z_{1}^{(4)}}&=\\
\frac{\partial z_{1}^{(4)}}{\partial a_{1}^{(3)}}&=\\
\frac{\partial a_{1}^{(3)}}{\partial z_{1}^{(3)}}&=\\
\frac{\partial z_{1}^{(3)}}{\partial a_{1}^{(2)}}&=\\
\frac{\partial a_{1}^{(2)}}{\partial z_{1}^{(2)}}&=\\
\frac{\partial z_{1}^{(2)}}{\partial W^{(1)}_{11}}&=\end{align}$$

- Multiplying to get:

$$\frac{\partial}{\partial W^{(1)}_{11}}E=$$

### (WIP) Backprop Algorithm
(put algorithm here with pseudo-code)

- (insert this in detailed section of overview with math included) After calculating gradient, Update weight:

$$W^{(k)}_{ij}\leftarrow W^{(k)}_{ij}-\alpha\frac{\partial}{\partial W^{(k)}_{ij}}E$$

(put proof here with WWTP: partial derivative of E for any Wl,i,j)

### General NN $$\nabla E$$ Calculation
- Additional Notation:
  - $$s_{k}$$ denotes number of neurons in layer $$k$$
- Final Goal = Calculate the overall gradient of the error:

$$\nabla E$$

- The gradient of the error = partial derivatives of error with respect to each weight:

$$\nabla E = \left(\frac{\partial}{\partial W^{(1)}_{11}}E,\ldots,\frac{\partial}{\partial W^{(k)}_{ij}}E,\ldots,\frac{\partial}{\partial W^{(n)}_{s_{n-1}s_{n}}}E\right)$$

# Jupyter Cheatsheet
- tab: allows you to complete variable names or list functions of a package within code cell
- shift + tab: lets you see function documentation, variable values
- (shift + tab)x2: allows you to see more in-depth documentation
- markdown + latex: $ or $$ and can insert latex
- enter: enter edit mode
- escape: enter command mode
- h: show all commands
- a: create cell above
- b: create cell below
- y: change to code cell
- m: change to markdown cell
- l: toggle line numbers
- d + d: delete cell
- escape + s : saves notebook
- shift + ctrl + p: enter command palette
- arrow keys: move around cells
- x: cut
- c: copy
- v: paste
- space: scroll down
- shift + space: scroll up
- %timeit function(): allows you to time function
- %%timeit: allows you to time entire cell (at top of cell)
- %pdb: turn on interactive debugger (q: turns it off)
- ([additional](http://ipython.readthedocs.io/en/stable/interactive/magics.html) magic commands)
- View > Cell Toolbar > Slideshow: bring up slide cell menu
- jupyter nbconvert notebook.ipynb --to slides: convert to slideshow from file
- jupyter nbconvert notebook.ipynb --to slides --post serve: convert to slideshow and immediately see

# Numpy Cheatsheet
- `x = v[None, :]` ~ add new dimension to array
