*Machine Learning*: "field of study that gives computers the ability to learn without being explicitly programmed." (Arthur Samuel, 1959)

*Well-posed Learning Problem*: "a computer program is said to LEARN from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E." (Tom Mitchell, 1998)

### Machine Learning algorithms:

- Supervised learning
- Unsupervised learning
- Others: reinforcement learning, recommender systems.

#### Supervised Learning

- Regression: predict continuous valued output.
- Classification: predict discrete valued output.

### Unsupervised Learning

- Cocktail party problem algorithm [Octave]:

~~~
[W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');
~~~

	- svd: Singular Value Decomposition
	- repmat: repeat matrix

# Linear Regression

## Model representation

- Training set of housing prices (Portland, OR)
	- $x$: size in squared feet (input variable / feature)
	- $y$: price in thou. of US dollars (output variable / target)
	- $m$: number of training examples

	- $(x,y)$: one training example
	- $(x^{(i)}, y^{(i)})$: $i$-th training example

	Training set -> Learning algorithm -> $h$ (hypothesis)

	- $h$: maps from $x$ to $y$.
	
	$ h_{\theta}(x) = \theta_0 + \theta_1 x $

		- Shorthand: $h(x)$
		- Univariate linear regression.

## Cost function

Hypothesis: $h(x) = \theta_0 + \theta_1 x$
- $\theta_i$'s: parameters
	- how to choose $\theta_i$'s?

- Optimization problem:
$ \underset{\theta_0, \theta_1}{min} J\big( \theta_0, \theta_1 \big) := \displaystyle\frac{1}{2m} \displaystyle\sum_{i=1}^{m} \bigg( h_{\theta} \big( x^{(i)} \big) - y^{(i)} \bigg)^{2}$
	- Squared error function, or mean squared error.
		- "The mean is halved as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term."

## Gradient Descent

Have some function $J(\theta_0, \theta_1)$ we want to minimize.
	- Start with some $\theta_0, \theta_1$;
	- Keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up at a minimum.

- Algorithm:
	repeat until convergence{
		$\theta_j := \theta_j - \alpha \displaystyle\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) $ (for $j = 0$ and $ j = 1$)
	}

- Correct: simultaneous update

temp0 $ := \theta_0 - \alpha \displaystyle\frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)$
temp1 $:= \theta_1 - \alpha \displaystyle\frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1)$
$\theta_0 := $ temp0
$\theta_1 := $ temp1

- $\alpha$: learning rate

- Applying Gradient Descent to Linear Regression

$\displaystyle\frac{\partial}{\partial \theta_{j}} J(\theta_0, \theta_1) = \displaystyle\frac{\partial}{\partial \theta_{j}} \displaystyle\frac{1}{2m} \displaystyle\sum_{i=1}^{m} \bigg( h_{\theta} \big( x^{(i)} \big) - y^{(i)} \bigg)^{2}$

	- for $j = 0$: $\displaystyle\frac{\partial}{\partial \theta_{j}} J(\theta_0, \theta_1) = \displaystyle\frac{1}{m} \displaystyle\sum_{i=1}^{m} \bigg( h_{\theta} \big( x^{(i)} \big) - y^{(i)} \bigg)$
	- for $j = 1$: $\displaystyle\frac{\partial}{\partial \theta_{j}} J(\theta_0, \theta_1) = \displaystyle\frac{1}{m} \displaystyle\sum_{i=1}^{m} \bigg( h_{\theta} \big( x^{(i)} \big) - y^{(i)} \bigg) \cdot x^{(i)}$
