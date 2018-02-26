
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
- $$m$$ : number of training samples
- $$X_{i}$$ : a specific train sample $$i$$
- $$n$$ : number of features
- $$w_{j}$$ : a specific feature weight
- $$x_{j}$$ : a specific input feature
- $$\hat{y}_{i}$$ : prediction for a specific train sample $$i$$

## Cross-Entropy Equations
- Calculate predictions:

$$\begin{align}\hat{y}_{i}&=\sigma(WX_{i}+b) \\
\hat{y}_{i}&=\sigma(\sum_{j=1}^{n}w_{j}x_{j}+b) \\
\hat{y}_{i}&=\sigma(w_{1}x_{1}+\ldots+w_{n}x_{n}+b)\end{align}$$

- Cross Entropy (2 classes):

$$E=-\frac{1}{m}\sum_{i=1}^{m}y_{i}\ln(\hat{y}_{i})+(1-y_{i})\ln(1-\hat{y}_{i})$$

- Cross Entropy (2 classes, with W = weights, b = bias):

$$E(W,b)=-\frac{1}{m}\sum_{i=1}^{m}y_{i}\ln(\sigma(WX_{i}+b))+(1-y_{i})\ln(1-\sigma(WX_{i}+b))$$

- Cross Entropy (n classes):

$$E=-\sum_{i=1}^{m}\sum_{j=1}^{n}y_{ij}\ln(\hat{y}_{ij})$$

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
E_{i}=-y_{i}\ln(\hat{y}_{i})-(1-y_{i})\ln(1-\hat{y}_{i})
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
\begin{align}\frac{\partial}{\partial w_{j}}E_{i}&=\frac{\partial}{\partial w_{j}}(-y_{i}\ln(\hat{y}_{i})-(1-y_{i})\ln(1-\hat{y}_{i})) &&(E_{i}\text{ formula)}\\
&= -y_{i}(\frac{\partial}{\partial w_{j}}(\ln(\hat{y}_{i})))-(1-y_{i})(\frac{\partial}{\partial w_{j}}(\ln(1-\hat{y}_{i})))\\
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
- $$s_{l}$$ : number of neurons in layer $$l$$
- $$L$$ : number of layers, including input layer
- $$z^{(l)}_{j}$$ : the output of the $$j^{\text{th}}$$ neuron in the $$l^{\text{th}}$$ layer before applying sigmoid function
- $$a^{(l)}_{j}$$ : the output of the $$j^{\text{th}}$$ neuron in the $$l^{\text{th}}$$ layer after applying sigmoid function
- $$\text{subscript}\ \ i$$ : for specific training sample

## NN Forward Propagation Equations

- Calculating NN predictions for train sample $$X_{i}$$:

$$\hat{y}_{i}=\sigma(W^{(L-1)}(\sigma(W^{(L-2)}(\ldots(\sigma(W^{(1)}X_{i}))))))$$

- Example when $$L=4$$:

$$\hat{y}_{i}=\sigma(W^{(3)}(\sigma(W^{(2)}(\sigma(W^{(1)}X_{i})))))$$

## NN Backpropagation
### Backprop Method Overview
- Perform Forward Propagation
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

$$\begin{align}
\frac{\partial}{\partial W^{(1)}_{11}}E &= \left(\frac{a_1^{(4)}-y}{a_1^{(4)}(1-a_1^{(4)})}\right) \\
&\phantom{0000} \cdot \left(a_1^{(4)}(1-a_1^{(4)})\right) \\
&\phantom{0000} \cdot \left(\left(W_{11}^{(3)} \cdot a_1^{(3)}(1-a_1^{(3)}) \cdot W_{11}^{(2)}\right) + \left(W_{21}^{(3)} \cdot a_2^{(3)}(1-a_2^{(3)}) \cdot W_{12}^{(2)}\right) + \left(W_{31}^{(3)} \cdot a_3^{(3)}(1-a_3^{(3)}) \cdot W_{13}^{(2)}\right)\right) \\
&\phantom{0000} \cdot \left(a_1^{(2)}(1-a_1^{(2)})\right) \\
&\phantom{0000} \cdot \left(a_1^{(1)}\right)
\end{align}$$

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
\frac{\partial E}{\partial W_{11}^{(1)}} = \frac{\partial E}{\partial a_1^{(4)}} \cdot \frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} \cdot \frac{\partial z_1^{(4)}}{\partial a_1^{(3)}} \cdot \frac{\partial a_1^{(3)}}{\partial z_1^{(3)}} \cdot \frac{\partial z_1^{(3)}}{\partial a_1^{(2)}} \cdot \frac{\partial a_1^{(2)}}{\partial z_1^{(2)}} \cdot \frac{\partial z_1^{(2)}}{\partial W_{11}^{(1)}} \\\\
$$

$$\begin{align}
\text{Incorrect because it only propagates error from }& a_1^{(4)}\to a_1^{(3)}\to a_1^{(2)}\to W_{11}^{(1)} \\
\text{And neglects error that also propagates from }& a_1^{(4)}\to a_2^{(3)}\to a_1^{(2)}\to W_{11}^{(1)} \\
\text{as well as from }& a_1^{(4)}\to a_3^{(3)}\to a_1^{(2)}\to W_{11}^{(1)} \\\\
\end{align}$$

$$\text{Correct use of chain rule:}$$

$$
\frac{\partial E}{\partial W_{11}^{(1)}} = \frac{\partial E}{\partial a_1^{(4)}} \cdot \frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} \cdot \frac{\partial z_1^{(4)}}{\partial a_1^{(2)}} \cdot \frac{\partial a_1^{(2)}}{\partial z_1^{(2)}} \cdot \frac{\partial z_1^{(2)}}{\partial W_{11}^{(1)}} \\\\
$$

$$\text{Calculating partial derivatives:}$$

$$\begin{align}
\frac{\partial E}{\partial a_1^{(4)}}&=\frac{\partial}{\partial a_1^{(4)}}(-y\ln(a_1^{(4)})-(1-y)\ln(1-a_1^{(4)}))\\
&=-y\cdot\frac{\partial}{\partial a_1^{(4)}}(\ln(a_1^{(4)})-(1-y)\cdot\frac{\partial}{\partial a_1^{(4)}}(\ln(1-a_1^{(4)}))\\
&=-y\cdot\frac{1}{a_1^{(4)}}\cdot 1-(1-y)\cdot\frac{1}{1- a_1^{(4)}}\cdot -1\\
&=\frac{a_1^{(4)}-y}{a_1^{(4)}(1-a_1^{(4)})} \\\\
\end{align}$$

$$\begin{align}
\frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} &= \frac{\partial }{\partial z_1^{(4)}}(\sigma(z_1^{(4)})) \\
&= \sigma(z_1^{(4)})(1-\sigma(z_1^{(4)})) &&(\sigma'(x)=\sigma(x)(1-\sigma(x)) \\
&= a_1^{(4)}(1-a_1^{(4)}) &&(a_1^{(4)}=\sigma(z_1^{(4)})) \\\\
\end{align}$$

$$\begin{align}
\frac{\partial z_1^{(4)}}{\partial a_1^{(2)}} &= \frac{\partial }{\partial a_1^{(2)}}(W_{11}^{(3)}a_1^{(3)} + W_{21}^{(3)}a_2^{(3)} + W_{31}^{(3)}a_3^{(3)}) \\
&= \frac{\partial }{\partial a_1^{(2)}}(W_{11}^{(3)}\sigma(z_1^{(3)}) + W_{21}^{(3)}\sigma(z_2^{(3)}) + W_{31}^{(3)}\sigma(z_3^{(3)})) \\
&= W_{11}^{(3)} \cdot \frac{\partial }{\partial a_1^{(2)}}(\sigma(z_1^{(3)})) + W_{21}^{(3)} \cdot \frac{\partial }{\partial a_1^{(2)}}(\sigma(z_2^{(3)})) + W_{31}^{(3)} \cdot \frac{\partial }{\partial a_1^{(2)}}(\sigma(z_3^{(3)})) \\
&= W_{11}^{(3)} \cdot \sigma(z_1^{(3)})(1-\sigma(z_1^{(3)})) \cdot \frac{\partial }{\partial a_1^{(2)}}(z_1^{(3)}) \\
&\phantom{0000} + W_{21}^{(3)} \cdot \sigma(z_2^{(3)})(1-\sigma(z_2^{(3)})) \cdot \frac{\partial }{\partial a_1^{(2)}}(z_2^{(3)}) \\
&\phantom{0000} + W_{31}^{(3)} \cdot \sigma(z_3^{(3)})(1-\sigma(z_3^{(3)})) \cdot \frac{\partial }{\partial a_1^{(2)}}(z_3^{(3)}) \\
&= W_{11}^{(3)} \cdot a_1^{(3)}(1-a_1^{(3)}) \cdot \frac{\partial }{\partial a_1^{(2)}}(z_1^{(3)}) \\
&\phantom{0000} + W_{21}^{(3)} \cdot a_2^{(3)}(1-a_2^{(3)}) \cdot \frac{\partial }{\partial a_1^{(2)}}(z_2^{(3)}) \\
&\phantom{0000} + W_{31}^{(3)} \cdot a_3^{(3)}(1-a_3^{(3)}) \cdot \frac{\partial }{\partial a_1^{(2)}}(z_3^{(3)}) \\
&= W_{11}^{(3)} \cdot a_1^{(3)}(1-a_1^{(3)}) \cdot \frac{\partial }{\partial a_1^{(2)}}(W_{11}^{(2)}a_1^{(2)} + W_{21}^{(2)}a_2^{(2)} + W_{31}^{(2)}a_3^{(2)} + W_{41}^{(2)}a_4^{(2)}) \\
&\phantom{0000} + W_{21}^{(3)} \cdot a_2^{(3)}(1-a_2^{(3)}) \cdot \frac{\partial }{\partial a_1^{(2)}}(W_{12}^{(2)}a_1^{(2)} + W_{22}^{(2)}a_2^{(2)} + W_{32}^{(2)}a_3^{(2)} + W_{42}^{(2)}a_4^{(2)}) \\
&\phantom{0000} + W_{31}^{(3)} \cdot a_3^{(3)}(1-a_3^{(3)}) \cdot \frac{\partial }{\partial a_1^{(2)}}(W_{13}^{(2)}a_1^{(2)} + W_{23}^{(2)}a_2^{(2)} + W_{33}^{(2)}a_3^{(2)} + W_{43}^{(2)}a_4^{(2)}) \\
&= W_{11}^{(3)} \cdot a_1^{(3)}(1-a_1^{(3)}) \cdot \left(\frac{\partial }{\partial a_1^{(2)}}(W_{11}^{(2)}a_1^{(2)}) + 0 + 0 + 0 \right) \\
&\phantom{0000} + W_{21}^{(3)} \cdot a_2^{(3)}(1-a_2^{(3)}) \cdot \left(\frac{\partial }{\partial a_1^{(2)}}(W_{12}^{(2)}a_1^{(2)}) + 0 + 0 + 0 \right) \\
&\phantom{0000} + W_{31}^{(3)} \cdot a_3^{(3)}(1-a_3^{(3)}) \cdot \left(\frac{\partial }{\partial a_1^{(2)}}(W_{13}^{(2)}a_1^{(2)}) + 0 + 0 + 0 \right) \\
&= \left(W_{11}^{(3)} \cdot a_1^{(3)}(1-a_1^{(3)}) \cdot W_{11}^{(2)}\right) + \left(W_{21}^{(3)} \cdot a_2^{(3)}(1-a_2^{(3)}) \cdot W_{12}^{(2)}\right) + \left(W_{31}^{(3)} \cdot a_3^{(3)}(1-a_3^{(3)}) \cdot W_{13}^{(2)}\right) \\\\
\end{align}$$

$$\begin{align}
\frac{\partial a_1^{(2)}}{\partial z_1^{(2)}} &= \frac{\partial }{\partial z_1^{(2)}}(\sigma(z_1^{(2)})) \\
&= \sigma(z_1^{(2)})(1-\sigma(z_1^{(2)})) \\
&= a_1^{(2)}(1-a_1^{(2)}) \\\\
\end{align}$$

$$\begin{align}
\frac{\partial z_1^{(2)}}{\partial W_{11}^{(1)}} &= \frac{\partial }{\partial W_{11}^{(1)}}(W_{11}^{(1)}a_1^{(1)} + W_{21}^{(1)}a_2^{(1)} + W_{31}^{(1)}a_3^{(1)}) \\
&= \frac{\partial }{\partial W_{11}^{(1)}}(W_{11}^{(1)}a_1^{(1)}) + 0 + 0 \\
&= a_1^{(1)} \\\\
\end{align}$$

$$\text{Using calculated partial derivatives and chain rule:}$$

$$\begin{align}
\frac{\partial E}{\partial W_{11}^{(1)}} &= \frac{\partial E}{\partial a_1^{(4)}} \cdot \frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} \cdot \frac{\partial z_1^{(4)}}{\partial a_1^{(2)}} \cdot \frac{\partial a_1^{(2)}}{\partial z_1^{(2)}} \cdot \frac{\partial z_1^{(2)}}{\partial W_{11}^{(1)}} \\\\
&= \left(\frac{a_1^{(4)}-y}{a_1^{(4)}(1-a_1^{(4)})}\right) \\
&\phantom{0000} \cdot \left(a_1^{(4)}(1-a_1^{(4)})\right) \\
&\phantom{0000} \cdot \left(\left(W_{11}^{(3)} \cdot a_1^{(3)}(1-a_1^{(3)}) \cdot W_{11}^{(2)}\right) + \left(W_{21}^{(3)} \cdot a_2^{(3)}(1-a_2^{(3)}) \cdot W_{12}^{(2)}\right) + \left(W_{31}^{(3)} \cdot a_3^{(3)}(1-a_3^{(3)}) \cdot W_{13}^{(2)}\right)\right) \\
&\phantom{0000} \cdot \left(a_1^{(2)}(1-a_1^{(2)})\right) \\
&\phantom{0000} \cdot \left(a_1^{(1)}\right) \ \ \ \ \blacksquare\\\\
\end{align}$$

### Calculating $$\frac{\partial}{\partial W^{(1)}_{21}}E$$
For the sample NN:

$$\begin{align}
\frac{\partial}{\partial W^{(1)}_{21}}E &= \left(\frac{a_1^{(4)}-y}{a_1^{(4)}(1-a_1^{(4)})}\right) \\
&\phantom{0000} \cdot \left(a_1^{(4)}(1-a_1^{(4)})\right) \\
&\phantom{0000} \cdot \left(\left(W_{11}^{(3)} \cdot a_1^{(3)}(1-a_1^{(3)}) \cdot W_{11}^{(2)}\right) + \left(W_{21}^{(3)} \cdot a_2^{(3)}(1-a_2^{(3)}) \cdot W_{12}^{(2)}\right) + \left(W_{31}^{(3)} \cdot a_3^{(3)}(1-a_3^{(3)}) \cdot W_{13}^{(2)}\right)\right) \\
&\phantom{0000} \cdot \left(a_1^{(2)}(1-a_1^{(2)})\right) \\
&\phantom{0000} \cdot \left(a_2^{(1)}\right)
\end{align}$$

#### Proof.

$$\text{Virtually identical proof as $\frac{\partial}{\partial W^{(1)}_{11}}E$,} \\
\text{Except for one partial derivative:}$$

$$\frac{\partial E}{\partial W_{11}^{(1)}} = \frac{\partial E}{\partial a_1^{(4)}} \cdot \frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} \cdot \frac{\partial z_1^{(4)}}{\partial a_1^{(2)}} \cdot \frac{\partial a_1^{(2)}}{\partial z_1^{(2)}} \cdot \frac{\partial z_1^{(2)}}{\partial W_{21}^{(1)}} \\\\
$$

$$\text{Calculating $\frac{\partial z_1^{(2)}}{\partial W_{21}^{(1)}}$:}$$

$$\begin{align}
\frac{\partial z_1^{(2)}}{\partial W_{21}^{(1)}} &= \frac{\partial }{\partial W_{21}^{(1)}}(W_{11}^{(1)}a_1^{(1)} + W_{21}^{(1)}a_2^{(1)} + W_{31}^{(1)}a_3^{(1)}) \\
&= 0 + \frac{\partial }{\partial W_{21}^{(1)}}(W_{21}^{(1)}a_2^{(1)}) + 0 \\
&= a_2^{(1)} \\\\
\end{align}$$

$$
\text{Reusing previously calculated partial derivatives,} \\
\text{and $\frac{\partial z_1^{(2)}}{\partial W_{21}^{(1)}}$, yields desired result} \ \ \ \ \blacksquare\\\\
$$

### Calculating $$\frac{\partial}{\partial W^{(3)}_{11}}E$$
For the sample NN:

$$\begin{align}
\frac{\partial}{\partial W^{(1)}_{21}}E &= \left(\frac{a_1^{(4)}-y}{a_1^{(4)}(1-a_1^{(4)})}\right) \cdot \left(a_1^{(4)}(1-a_1^{(4)}\right) \cdot \left( a_1^{(3)}\right)
\end{align}$$

#### Proof.

$$\text{For simplicity, Let:}$$

$$\begin{align}
x_1=a_1^{(1)},\ x_2&=a_2^{(1)},\ x_3=a_3^{(1)}, \\
\hat{y}&=a_1^{(4)}
\end{align}$$

$$\text{Equations from sample NN:}$$

$$\begin{align}
z_1^{(4)} &= W_{11}^{(3)}a_1^{(3)} + W_{21}^{(3)}a_2^{(3)} + W_{31}^{(3)}a_3^{(3)} \\
a_1^{(4)} &= \sigma(z_1^{(4)}) \\
E &= -y\ln(a_1^{(4)})-(1-y)\ln(1-a_1^{(4)})\\\\
\end{align}$$

$$\text{Using chain rule:}$$

$$
\frac{\partial E}{\partial W_{11}^{(3)}} = \frac{\partial E}{\partial a_1^{(4)}} \cdot \frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} \cdot \frac{\partial z_1^{(4)}}{\partial W_{11}^{(3)}} \\\\
$$

$$\text{From proof for $\frac{\partial}{\partial W^{(1)}_{11}}E$:}$$

$$
\frac{\partial E}{\partial a_1^{(4)}}=\frac{a_1^{(4)}-y}{a_1^{(4)}(1-a_1^{(4)})} \\\\
$$

$$
\frac{\partial a_1^{(4)}}{\partial z_1^{(4)}} = a_1^{(4)}(1-a_1^{(4)}) \\\\
$$

$$\begin{align}
\frac{\partial z_1^{(4)}}{\partial W_{11}^{(3)}} &= \frac{\partial }{\partial W_{11}^{(3)}}(W_{11}^{(3)}a_1^{(3)} + W_{21}^{(3)}a_2^{(3)} + W_{31}^{(3)}a_3^{(3)}) \\
&= \frac{\partial }{\partial W_{11}^{(3)}}(W_{11}^{(3)}a_1^{(3)}) + 0 + 0 \\
&= a_1^{(3)} \\\\
\end{align}$$

$$
\frac{\partial E}{\partial W_{11}^{(3)}} = \left(\frac{a_1^{(4)}-y}{a_1^{(4)}(1-a_1^{(4)})}\right) \cdot \left(a_1^{(4)}(1-a_1^{(4)}\right) \cdot \left( a_1^{(3)}\right) \ \ \ \ \blacksquare\\\\
$$

### (WIP) Backprop Algorithm Pseudo-Code
#### Hyperparameters
- $$M=$$ number of training examples
- $$m=$$ batch size (assuming $$M$$ divisible by $$m$$)
- $$E=$$ number of epochs to train for
- $$L=$$ number of NN layers
- $$n=$$ number of features per train sample
- $$c=$$ number of classes
- $$\alpha=$$ learning rate
- $$[s_1,s_2,\ldots,s_l,\ldots,s_L] = $$ number of neurons per layer list

#### Notation (Pythonic)
- same notation as NN Notation
- `np.dot()` : numpy dot product of vectors
- `np.ndarray.T` : numpy matrix transpose
- `*` : numpy element-wise multiplication
- `np.ndarray.shape` : numpy array shape
- `sigmoid()` : sigmoid function

#### Training Data (Pythonic)
- All training samples: $$X, Y$$ numpy array
- Each training sample has features: $$X[i]=[[x_1,x_2,\ldots,x_n]]$$
- Each training sample has labels: $$Y[i]=[[y_1,y_2,\ldots,y_c]]$$

#### Object Attributes (Pythonic)
- Let $$len(W)=L$$
- for $$l$$ in $$range(1,(L-1)+1)$$
  - Let $$W[l].shape=(s_{l+1},s_l)$$
    - Implies $$W[l][j].shape=(s_l,)$$
    - Implies $$W[l][j]=\text{vector of length}\ s_l$$
  - Let $$grad\_sum\_W[l].shape=(s_{l+1},s_l)$$
- Let $$X.shape=(M,n)=(M,s_1)$$
  - Implies $$X[i].shape=(s_1,)$$
  - Implies $$X[i]=\text{vector of length}\ s_1$$
- Let $$Y.shape=(M,c)=(M,s_L)$$
  - Implies $$Y[i].shape=(s_L,)$$
  - Implies $$Y[i]=\text{vector of length}\ s_L$$
- for $$l$$ in $$range(1,L+1)$$:
  - Let $$a[l].shape=(1,s_l)$$
  - Let $$\delta[l].shape=(1,s_l)$$

#### Pseudo-Code (Pythonic)
- for $$l$$ in $$range(1,(L-1)+1)$$
  - Let $$grad\_sum\_W[l]=0$$
- for $$e$$ in $$range(1,E+1)$$:
  - for $$M_i, (x,y)$$ in $$enumerate(zip(X,Y))$$:
    - Let $$a[1]=x[None,:]$$
    - for $$l$$ in $$range(2,L+1)$$:
      - Compute $$a[l]=sigmoid(np.matmul(a[l-1],W[l-1].T))$$
    - Compute $$\delta[L]=a[L]-y$$
    - for $$l$$ in $$range(2,(L-1)+1)$$:
      - Compute $$\delta[l]=np.matmul(\delta[l+1],W[l])*a[l]*(1-a[l])$$
    - for $$l$$ in $$range(1,(L-1)+1)$$
      - Compute $$grad\_sum\_W[l]+=np.matmul(\delta[l+1].T,a[l])$$
    - if $$(M_i+1)\ \%\ m == 0$$:
      - for $$l$$ in $$range(1,(L-1)+1)$$
        - Compute $$W[l]=W[l]-\alpha\frac{1}{m}grad\_sum\_W[l]$$
        - Let $$grad\_sum\_W[l]=0$$

#### Notation (Mathematical)
- same notation as NN Notation
- $$A^{T}$$ : matrix transpose
- $$AB$$ : matrix multiplication of matrices $$A$$ and $$B$$
- $$A \circ B$$ : element wise multiplication of matrices $$A$$ and $$B$$
- $$\sigma()$$ : sigmoid function

#### Training Data (Mathematical)
- All training samples: $$(X_1,Y_1),(X_2,Y_2),\ldots,(X_M,Y_M)$$
- Each training sample has features: $$(x_1,x_2,\ldots,x_n)$$
- Each training sample has labels: $$(y_1,y_2,\ldots,y_n)$$

#### Pseudo-Code (Mathematical)
- $$\forall\ l\in\{1,\ldots,L-1\},$$ Let $$\frac{\partial}{\partial W^{(l)}}E = 0$$
- For every epoch:
  - For every train sample $$(X_i,Y_i)$$ in $$(X_1,Y_1),\ldots,(X_M,Y_M)$$:
    - Let $$a_1^{(1)}=x_1,\ a_2^{(1)}=x_2,\ \ldots,\ a_{s_1}^{(1)}=x_n$$
    - $$\forall\ l\in\{2,\ldots,L\},$$ Compute $$a^{(l)}=\sigma(a^{(l-1)}W^{(l-1)T})$$
    - Compute $$\delta^{(L)}=a^{(L)}-Y_i$$
    - $$\forall\ l\in\{2,\ldots,L-1\},$$ Compute $$\delta^{(l)} = \delta^{(l+1)}W^{(l)} \circ a^{(l)} \circ (1-a^{(l)})$$
    - Update $$\frac{\partial}{\partial W^{(l)}}E = \frac{\partial}{\partial W^{(l)}}E+\delta^{(l+1)T}a^{(l)}$$

#### Proof.

(put algorithm here with pseudo-code)

- (insert this in detailed section of overview with math included) After calculating gradient, Update weight:

$$W^{(k)}_{ij}\leftarrow W^{(k)}_{ij}-\alpha\frac{\partial}{\partial W^{(k)}_{ij}}E$$

(put proof here with WWTP: partial derivative of E for any Wl,i,j)

$$\nabla E = \left(\frac{\partial}{\partial W^{(1)}_{11}}E,\ldots,\frac{\partial}{\partial W^{(l)}_{ij}}E,\ldots,\frac{\partial}{\partial W^{(L)}_{s_{L-1}s_{L}}}E\right)$$

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
- np array shapes:
  - `(rows, columns)`
  - `(layers, rows, columns)`
- `x = v[None, :]` : add new dimension to array

# LivingProgram Notes Convention
* When dealing with pseudo-code:
  * Pythonic Notation: python code, subscripts for variables allowed
  * Mathematical Notation: pure math
