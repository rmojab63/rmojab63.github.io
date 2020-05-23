---
layout: presentation
htmllang: en
title: Estimation Framework
---

<section data-markdown data-separator="^SlideNext$" data-separator-vertical="^SlideNextV$">

# Estimation Frameworks

Ramin Mojab, 2018


SlideNext

## Descriptive Statistics
- Descriptive Statistics consists of providing visual or quantitative summaries about a set of data (or observations).
- We might use simple plots or calculate percentages, means, variances, etc.
- We generally provide information about two types of measures: central tendency and dispersion.
- Where does the sample come from? What is the population? How is data distributed there? What happens to the mean or variance if we double the size of the sample?
  + We do not deal with such questions in this subject.

SlideNext

## Inferential Statistics
- Assuming that the data is just a sample of a larger population, **inferential statistics** can help us infer properties of the population.

SlideNextV

## Logical Reasoning
- Do you like to reach a logically certain conclusion about something? well, you must be familiar with the following forms:
- **Deduction**: All humans are mortal, you are a human, therefore, you are mortal.
  + The truth of the premises guarantees the truth of the conclusion.
  + Deductive arguments are either _valid_ or _invalid_. Valid arguments are _sound_ if premises are true.
- **Induction**: You are writing with your right hand, therefore, you are a right-handed man.
  + Even in its best cases, the truth of the premises does not guarantee the truth of the conclusion.
  + inductive arguments are either _strong_ or _weak_.

SlideNextV

## The Problem of Induction
- Does inductive reasoning lead to knowledge?
- There was a time when everyone in Europe believed that all swans are white, because "all swans we have seen are white, and, therefore, all swans are white", until someone saw a black one.
- Karl Popper argued that science does not use induction.
  + The main role of observation in science is in falsification.
  + Science should NOT search for theories that are probably true, instead, it should seek for theories that are falsifiable, but all attempt to falsify them have failed so far.

SlideNextV

## Bayes and Popper
- Popperian science: a hypothesis is made (based on some rules,) and then a deductive logic is used to falsify it.
  + It can never be accepted, but it can be rejected (that is, falsified).
- Bayesian science: based on an inductive approach, starts from a prior belief, uses some data, moves toward a posterior belief.

SlideNext

## A Statistical Model
- Statistical modelling is a formal representation of a theory and is an essential part of any statistical inference.
- A Statistical Model is a set of _statistical assumptions_ about a population. It consists of:
1. Data (i.e., a probability space without the probablity measure part);
1. a **set of** probability distributions.
- A distribution almost always contains parameters.
  + The set of probability distributions of the model can be built using such a set; i.e., a probability distribution is assignsed to each parameter point.
- As a result, a statistical model consists of _unknown_ parameters, i.e., **Statistical Parameters**.

SlideNextV

## Data
- Data is an essential part of a statistical model.
- This part of the story starts from  **outcome**, **event** and **experiment**.
  + An outcome is a possible result of an experiment.
  + An event is a set of outcomes.
  + An experiment can be infinitely repeated and has well-defined events.
- **Probability space** models an expriment mathematically. 
- It is a triple such as $(\Omega,\mathcal{F},P)$

SlideNextV
## Sample Space ($\Omega$)
- The non-empty set of all possible outcomes of a experiment.
- tossing a coin: {head, tail}.
- tossing two coins: {(head, tail), (head, head), (tail, tail),(tail, head)}.
- tossing a single six-sided dice: {1, 2, 3, 4, 5, 6}
- Height of a man (cm): $\mathbb{R}\_{+}$ 

SlideNextV

## Events ($\mathcal{F}$)
- As explained before, an event is a set of outcomes.
  + $\mathcal{F} \subseteq 2^{\Omega}$
- However, mathematically, this set must be a $\sigma$-algebra on the sample space.
- The following conditions must hold:
  + $\Omega \in \mathcal{F}$;
  + If $A\in \mathcal{F}$, then $A'\in \mathcal{F}$;
  + If $A_i\in \mathcal{F}$ for $i=1,2,...$, then $(\underset{i=1}{\overset{\infty}{\cup}} A\_i)\in \mathcal{F}$;

SlideNextV

## Probability Measure ($P$)
- This is a function on $\mathcal{F}$ (i.e., $P: \mathcal{F}\rightarrow [0,1]$) such that:
  + $P(\Omega)=1$;
  + $P(\underset{i=1}{\overset{\infty}{\cup}}A\_i)=\underset{i=1}{\overset{\infty}{\sum}}P(A\_i)$.


SlideNext

## How to Approximate the Unknown Parameters?
- What do we mean by _unknown_?!
  + We don't know the exact value; however,
  + The model might contain additional information about them (i.e., prior distributions).
- More information comes form the _events_.
  + _Measured Empirical Data_; _Sample Data_; _Observations_
- An **Estimator** approximates the statistical parameter using the measured empirical data.
- It is a _sample statistic_.
  + which means it can be computed by using the sample data.

SlideNext

## An Example
- Assume that we are interested in the _height of a 33 years old man_.
- The population is relatively large.
- We need some statistical assumptions about the population (i.e., a statistical model).
- Such assumptions are the first building blocks in our reasoning.
- We need them. We cannot infer conclusions from nothing. Of course;
  + They must be reasonable;
  + They must be carefully chosen;
  + They must be checked.
- For the current example and _for simplicity_, we choose **Normal Model**.
  + You might criticize the possibility of negative values.

SlideNext

## Normal Model
- We assume that data arise from a univariate normal distribution.
- i.e., the model has the following set of distributions;

$$
\\{\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}:~\mu\in  \mathbb{R},~\sigma\in  \mathbb{R}_{+}\\}
$$

- data (i.e., a set such as $\\{x_i\\}$ for $i=1,...,N$) comes from the events.
- Prior distributions might exist, too.
- The question is about estimation, i.e., choosing a subset of the set of the distributions.

SlideNextV

## Normal Distribution

- Probably the most important probability distributions.
- why?!
- Central Limit Theorem
  + roughly speaking, the average of independent random variables will converge to a normal distribution, as the number of observations increases
- Many things follow its rules!
  + Again, due to the central limit theorem
  + If an event is the sum of other (independent) events, it is distributed normally.
- It is also called _Gaussian_ distribution.
- It was discovered by Carl Friedrich Gauss (1777-1855).
- A version of it is called _Normal Standard_ or _z distribution_.

SlideNextV

## Probability Density Function

$$
f(x)  =  \frac{1}{\sqrt{2\pi}\sigma}  \cdot  e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}  
$$

- $x,\mu,\sigma  \in  \mathbb{R},  \sigma> 0$

- mean: $\mu$
- variance: $\sigma^2$
- notation: $\mathcal{N}(\mu,\sigma^2)$

- zero variance?! not a traditional function; It is zero everywhere except $x=\mu$ and its integral equals one!

SlideNextV

## Bell Curve
![Bell Curve Image](../../../assets/svgs/distributions/bellcurve.svg)

SlideNextV

## Density?!

- Well, we are interested in areas under portions of a normal distribution.
  + why?! For example, we need them for hypothesis testing
  + They Can be computed by using integral calculus. Of course, there is no exact _closed form_ expression for the integral over an arbitrary range.
  + Two general useful rules:
- 68% of the area is within one standard deviation of the mean.
- 95% of the area is within 1.96 standard deviations of the mean.

SlideNextV

![Normal 1](../../../assets/svgs/distributions/normal1.svg)

SlideNextV

![Normal 2](../../../assets/svgs/distributions/normal2.svg)

SlideNextV

![Normal 3](../../../assets/svgs/distributions/normal3.svg)

SlideNextV

## Cumulative Density Functions

$$F(x)=\frac{1}{2}  [1+\operatorname{erf}(\frac{x-\mu}{\sigma\sqrt{2}})]$$

SlideNextV

![CDF Normal 1](../../../assets/svgs/distributions/normalCDF1.svg)

SlideNextV

![CDF Normal 2](../../../assets/svgs/distributions/normalCDF2.svg)

SlideNextV

![CDF Normal 3](../../../assets/svgs/distributions/normalCDF3.svg)

SlideNextV

![CDF Normal 4](../../../assets/svgs/distributions/normalCDF4.svg)

SlideNext

# Gamma Model
- Alternatively, you might choose the following set of distributions;
$$
\\{\frac{(\frac{\nu}{2\delta})^{\frac{\nu}{2}}}{\Gamma(\frac{\nu}{2})}x^{\frac{\nu-2}{2}}e^{-\frac{\nu x}{2\delta}}
:~\alpha,\beta\in  \mathbb{R}_{+}\\}
$$
- $x>0$
- mean: $\delta$
- variance: $\frac{2\delta^2}{\nu}$

SlideNextV
# Another Parameterization
- Let $\beta=\frac{\nu}{2\delta}$ and $\alpha=\frac{\nu}{2}$:
$$
\\{\frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta  x}:~\alpha,\beta\in  \mathbb{R}_{+}\\}
$$

- $\alpha$ is the _shape_ parameter. It is not _location_ parameter, which simply shifts the distribution.

- $\beta$ is the _rate_ parameters. A larger value means the distribution will be more concentrated. The smaller value means it will be more spread out.


SlideNextV

# PDF
$$
f(x)  =  \frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta  x}
$$

- $x,\alpha,\beta  > 0$

- mean: $\frac{\alpha}{\beta}$
- variance: $\frac{\alpha}{\beta^2}$
- notation: $G(\alpha,\beta)$

- Note that this distribution has another parameterization with _scale_ parameter: $\theta=\frac{1}{\beta}$.

SlideNextV

![Gamma 1](../../../assets/svgs/distributions/gamma1.svg)

SlideNextV

![Gamma 2](../../../assets/svgs/distributions/gamma2.svg)

SlideNextV

![Gamma CDF 1](../../../assets/svgs/distributions/gammaCDF1.svg)

SlideNextV

# Inverse-Gamma
- In Bayesian literature, you might encounter with the inverse of gamma distribution.
$$
f(x)=f(\frac{1}{x})|-\frac{1}{x^2}|=\frac{\beta^{\alpha}}{\Gamma(\alpha)}(\frac{1}{x})^{\alpha-1}e^{-\beta\frac{1}{x}}\frac{1}{x^2}
$$
$$
f(x)  =  \frac{\beta^{\alpha}}{\Gamma(\alpha)}\frac{1}{x^{\alpha+1}}e^{-\frac{\beta}{x}}
$$
- $\beta$ plays the _scale_ role, therefore, its better to use another notation, e.g., $\theta$.

SlideNext

# Level of Assumptions

- Assuming a specific type of distribution might be controversial.
- The model of the previous example is parametric.
  + i.e., there is a finite number of parameters.
- We can relax some assumptions by moving toward non-parametric or semi-parametric estimation methods.
  + for such methods, the parameter set of the model is infinite dimensional.
- E.g., in the preceding example and (maybe) as a _more_ reasonable assumption, we might choose a set of distributions, in which their means are between 160 and 190.
  + Loosely speaking, a larger set of distributions.
- Fewer assumptions? Robust Conclusions? Why not?!

SlideNextV

# Level of Assumptions (continued)
- Two extreme cases:
  + no assumption, which means no conclusion.
  + assuming a specific number, a set of size one, therefore, a specific conclusion.

SlideNext

## The Trade-off

![Estimation Framework](../../../assets/svgs/estimationframework.svg)

SlideNext

## Degree of Subjectiveness

- A critical question:
  + Is it acceptable to use **non-data-based information** (i.e., personal opinions, interpretations, points of view, emotions, judgment) in the statistical inference?
- Before giving an answer, Note that objective inference is fact-based, measurable and observable.
- Anyway, if your answer is positive, you can move from classical (or frequentist) approaches toward Bayesian approach.
- Bayesian approach is more realistic and pragmatic. In fact, some problems cannot be tackled without subjective judgments.
- However, classic approaches are more theory-based.

SlideNext

## Another Trade-off

![Estimation Framework 2](../../../assets/svgs/estimationframework2.svg)

SlideNext

## Another Arena for Comparison: Estimation properties
- How much error is expected in very large samples?
  + "If you can’t get it right as $n$ goes to infinity, you shouldn’t be in this business." (Granger, from Wooldridge, 2016, p. 150)
- Which one uses the data most efficiently?
  + Should we compare the efficiency of two estimators that are based on different levels of assumptions?
  + As implied before, information is valuable.
  + "The best parametric estimator will generally outperform the best semi-parametric estimator." (Greene, 2002, p. 425)

SlideNext

## Maximum Likelihood Estimator
- Given the statistical model (observations and the set of univariate normal distributions), MLE selects a member for which the joint probability distribution of the observations is maximized.
$$
(\hat{\mu,}\hat{\sigma}^2)\in  \\{\underset{\mu\in  \mathbb{R},\sigma\in\mathbb{R}_{+}}{\text{arg max}}L(\mu,\sigma^2|x_1,...,x_n)\\}
$$
- The function $L(.)$ is called Likelihood function.
- The answer is the same if we use logarithm of $L(.)$

SlideNextV

## Likelihood Function
- The function $L(.)$ is just a different interpretation of the joint probability distribution of the observations.
$$
L(\mu,\sigma^2|x_1,...,x_N)=f(x_1,...,x_N|\mu,\sigma^2)
$$
- The joint distribution is continuous in our example, we can simply use differentiation.
- Since the differentiation is taken with respect to the parameters, we need a function in which the independent variables are the parameters.
- This is not the case with j.p.d. Therefore, Likelihood Function is used.

SlideNextV

## MLE Estimator 
- Assuming that observations are independent from each other,
$$
f(x_1,...,x_N|\mu,\sigma^2)=\underset{i=1}{\overset{N}{\Pi}}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\frac{x_i-\mu}{\sigma})^2}
$$
Do some math and you will find two familiar formulas,
$$
\begin{cases}
\hat{\mu}_N=\frac{\underset{i=1}{\overset{N}{\sum}}  x_i}{N}  \\\\
\hat{\sigma}_N^2=\frac{\underset{i=1}{\overset{N}{\sum}}(x_i-\hat{\mu})^2}{N}
\end{cases}
$$
These are the MLEs. Of course, you should check the second derivatives too.

SlideNextV

##  MLE Estimator (continued)
$$
\text{FOC}:\begin{cases}
\frac{\partial{lnL}}{\partial{\mu}}=\frac{1}{\hat{\sigma}^2}\underset{i=1}{\overset{N}{\sum}}(x_i-\mu)=0  \\\\
\frac{\partial{lnL}}{\partial{\sigma^2}}=\frac{N}{2\sigma^4}+\frac{1}{\hat{\sigma}^2}\underset{i=1}{\overset{N}{\sum}}(x_i-\mu)^2=0
\end{cases}
$$

$$
\text{Hessian}:  \mathbf{H}=
\begin{bmatrix}
\frac{-N}{\sigma^{2}}  & -\frac{1}{\sigma^4}\underset{i=1}{\overset{N}{\sum}}(x_i-\mu)  \\\\
-\frac{1}{\sigma^4}\underset{i=1}{\overset{N}{\sum}}(x_i-\mu)&\frac{N}{2\sigma^4}-\frac{1}{\sigma^6}\underset{i=1}{\overset{N}{\sum}}(x_i-\mu)^2
\end{bmatrix}
$$

- substituting the FOC results in a diagonal matrix with negative elements (i.e., a negative definite matrix)


SlideNextV

## Properties
- Given a MLE such as $\hat{\boldsymbol{\theta}}_N$ for $\boldsymbol{\theta}  \in  \boldsymbol{\Theta}$, we can prove that
$$
\sqrt{N}(\hat{\boldsymbol{\theta}}_N-\boldsymbol{\theta})\overset{d}{\rightarrow}N(\mathbf{0},\mathbf{I}^{-1})
$$
- The variance is the inverse of the Fisher Information matrix.
- We can also prove that for a consistent and asymptotically normally distributed estimator such as $\tilde{\boldsymbol{\theta}}_N$,
$$
\text{plim var}(\tilde{\boldsymbol{\theta}}_N)  \ge  \mathbf{I}^{-1}
$$
- Therefore, MLE is consistent, asymptotically normally distributed, and asymptotically efficient.

SlideNextV

## Fisher Information
- The observed Fisher information matrix is negative of the Hessian matrix of the log-likelihood function.
- The expected value of the observed Fisher information is the Fisher information.
- For our example,
$$
\mathbf{I}=-E(\mathbf{H})=\begin{bmatrix}
\frac{N}{\sigma^{2}}  & 0\\\\
0&\frac{2\sigma^4}{N}
\end{bmatrix}
$$
Note that $E(x_i-\mu)=0$ and $E(x_i-\mu)^2=\sigma^2$

SlideNext


## Bayes Estimator
- Despite their different philosiphical background, a part of the discussion is similar to the MLE case.
  + We need the joint p.d.f.
- Assuming that data points are independent from each other, we can derive a similar expression for the joint probability distribution.
- In this context, we know something _non-data-based_ about the parameters. Assume that they are presented in the following (prior) distribution, i.e., normal-gamma distribution:
$$
(\mu,\sigma^{-2})\sim  \text{NG}(\underline{\mu},\underline{\omega},\underline{\alpha},\underline{\beta}),  \quad
\underline{\mu}\in  \mathbb{R},~\underline{\omega},\underline{\delta},\underline{\nu}\in  \mathbb{R}\_{+}.
$$
- There are four _prior hyperparameters_.

SlideNextV

## A Prior
$$
\sigma^{-2}  \sim  G(\underline{\delta},  \underline{\nu}),  \quad\mu|  \sigma^2  \sim  N(\underline{\mu},  \underline{\omega}\sigma^2)
$$
- The first distribution is unconditional and therefore calibration is straightforward.
- The other is conditional, which means, your belief about $\mu$ is not independent from the value of $\sigma^2$.
  + Can't we use the more convenient unconditional distribution in this case?
  + We can, however, there will be no analytical solution, just numerical ones. (This is where Gibbs or Metropolis-Hastings samplers are used).
  + In other words, there is a trade-off between the simplicity of the prior and the simplicity of the calculations.
- Another point: Can we use a normal distribution for the variance?
  + no, the variance cannot be negative.

SlideNextV

## The Posterior
- Recall the Bayes theorem,
$$
P(A|D)=\frac{P(D|A)P(A)}{P(D)},\quad  P(D)\ne0
$$
- For current application, $A$ is the parameters and $D$ is _data_.
  + $P(A|D)$: the posterior; the probability of observing any set of parameters, if a specific set of data is observed.
  + $P(D|A)$ the joint distribution function of the observations, or the likelihood function.
  + $P(A)$ the prior.
  + $P(D)$ the marginal distribution of data.
- We are actually interested in $P(D|A)P(A)$ part. $P(D)$ is not a function of the parameters and $P(A|D)  \propto  P(D|A)P(A)$.

SlideNextV

## The Posterior (continued)
$$
P(\mu,\sigma^{-2}|x_1,...,x_N)  \propto  f(x_1,...,x_N|\mu,\sigma^2)\times  NG(\underline{\mu},\underline{\omega},\underline{\delta},\underline{\nu})
$$
We can show that,
$$
(\mu,\sigma^{-2}|x_1,...,x_N)  \sim  NG(\overline{\mu},\overline{\omega},\overline{\delta},\overline{\nu})
$$
where,
$$
\begin{cases}
\overline{\omega}=(\underline{\omega}^{-1}+N)^{-1}\\\\
\overline{\mu}=\overline{\omega}(\underline{\omega}^{-1}\underline{\mu}+N\hat{\mu})\\\\
\overline{\nu}=\underline{\nu}+N\\\\
\overline{\delta}^{-1}=\frac{\underline{\nu}}{\underline{\nu}+N}\underline{\delta}+\frac{N-1}{\underline{\nu}+N}s^2+\frac{1}{\underline{\nu}+N}(\hat{\mu}-\underline{\mu})^{2}[\underline{\omega}+N^{-1}]^{-1}
\end{cases}
$$
SlideNextV
## Some Notes
- The sum of weights in $\overline{\beta}$ equals $1$.
- Large number of observations:
$$N\rightarrow  \infty  \Rightarrow  \overline{\omega}\rightarrow  0  \Rightarrow  \begin{cases}
\overline{\mu}=\hat{\mu}\\\\
\overline{\delta}=s^{-2}
\end{cases}
$$
- High prior uncertainty: $\underline{\omega}  \rightarrow  \infty  \Rightarrow  \overline{\mu}=\hat{\mu}$
- Prior certainty: $\underline{\omega}  \rightarrow  0  \Rightarrow  \overline{\mu}=\underline{\mu}$
- Equal weights: $\underline{\omega}=\frac{1}{N}  \Rightarrow  \overline{\mu}=\frac{\underline{\mu}+\hat{\mu}}{2}$

SlideNextV
## The Estimator
- The preceding formula is a _distribution_. How about the estimator?
- A natural choice is the mean of the posterior distribution.
  + It minimizes the _mean square error_, (therefore the name, _minimum MSE estimator_).
- mode, median, and other quantiles have their own optimization logic.

SlideNextV

## Properties
- Loosely speaking, for large $N$, P(D|A) (i.e., the likelihood) is the dominant part of the posterior distribution.
- The posterior is asymptotically normally distributed, and minimum MSE estimator is the same as MLE.
 
SlideNext
 
## Regression Model
- Consider the following set of distributions:
$$
\\{\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}(\frac{y-a-b x}{\sigma})^2}: \; a,b\in\mathbb{R},\;\sigma\in\mathbb{R}\_{+}\\}
$$
- Letting $\epsilon=y-a-b x$, we can see this is a normal model for $\epsilon$, in which
  + $E(\epsilon)  = 0$; and
  + $var(\epsilon)=\sigma^2$.
- We generally use the following equation to describe this model:
$$
y=a+b x+\epsilon,  \;  \epsilon\sim  N(0,\sigma^2)
$$
- This is known as a **simple linear regression model**.
- Note that the model can be semi-parametric, i.e.,
$$
y=a+b x+\epsilon,  \;  E(\epsilon)=0,\;var(\epsilon)=\sigma^2
$$

SlideNextV

## Theoretical Model
- Where does $y, x, a, b$, and the linear relationship come from?
- Let $y$ be a person’s wage (measured in dollars per hour), and $x$ be her observed years of education.
- Assume that there is a theory which points to a mathematical equation that describes a linear relationship between these two variables;
$$
y=a+b x,  \;  a,b\in  \mathbb{R}
$$
- Note that the theory might restrict the parameters; e.g., $b>0$.

SlideNextV

## Theoretical Model vs. Regression Model
- _Formal_ theoretical modeling is a good starting point for empirical analysis, but it is not essential.
  + It might rely entirely on intuition or common sense.
- The Regression model is a generalization of the theoretical model:
  + The form of the functions must be specified in a regression model.
  + The choice of the variables in a regression model is determined by data considerations, as well as theory.
  + In a regression model, we should deal with the _disturbance term_.

SlideNext

## MLE (continued)
- Given data (i.e., a set such as $\\{(y_i,x_i)\\}$ for $i=1,...,N$) and similar to the previous discussion, we can calculate the MLE estimators.
- Let $\hat{e}_i=x_i-\hat{a}-\hat{b}x_i$;

$$
\text{FOC}:\begin{cases}
\frac{\partial{lnL}}{\partial{a}}=\frac{1}{\hat{\sigma}^2}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i=0  \\\\
\frac{\partial{lnL}}{\partial{b}}=\frac{1}{\hat{\sigma}^2}\underset{i=1}{\overset{N}{\sum}}x_i\hat{e}_i=0  \\\\
\frac{\partial{lnL}}{\partial{\sigma^2}}=\frac{N}{2\sigma^4}+\frac{1}{\hat{\sigma}^2}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i^2=0
\end{cases}
$$

SlideNextV
## MLE (Observed Hessian)

$$
\text{Hessian}:  \mathbf{H}=
\begin{bmatrix}
\frac{-N}{\sigma^{2}}  & -\frac{1}{\sigma^4}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i  \\\\
-\frac{1}{\sigma^4}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i&\frac{N}{2\sigma^4}-\frac{1}{\sigma^6}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i^2
\end{bmatrix}
$$

- Use FOCs to show this matrix is diagonal and negative definite at the solution.

SlideNextV

## The MLE Estimator
- Let $\bar{z}_N=\frac{\underset{i=1}{\overset{N}{\sum}}  x_i}{N}$;
$$
\begin{cases}
\hat{a}_N =  \bar{y}_N-\hat{b}_N\bar{x}_N  \\\\
\hat{b}_N =\frac{\underset{i=1}{\overset{N}{\sum}}(x_i-\bar{x}_N)(y_i-\bar{y}_N)}{\underset{i=1}{\overset{N}{\sum}}(\hat{x}_i-\bar{x}_N)^2}  \\\\
\hat{\sigma}_N^2=\frac{\underset{i=1}{\overset{N}{\sum}}\hat{e}_i^2}{N}  \\\\
\end{cases}
$$
- Apart from the mathematics and the formula, other general results of the normal model are valid.

SlideNextV
## Multiple Linear Regression

- The generalization for manipulating more variables from theory is straight-forward.
$$
y=  \mathbf{x}'  \boldsymbol{\mu}  +  \epsilon,  \quad  \epsilon\sim  N(0,\sigma^2)
$$
- $\mathbf{x},\boldsymbol{\mu}:k\times 1$
- Let $\hat{e}=y-\mathbf{x}'  \boldsymbol{\mu}$; MLE:
$$
\begin{cases}
\hat{ \boldsymbol{\mu}}_N =(\mathbf{X'X})^{-1}\mathbf{X'y}  \\\\
\hat{\sigma}_N^2=\frac{\mathbf{\hat{e}'\hat{e}}}{N}
\end{cases}
$$

SlideNextV

## Matrix Form
- The following set of observations is used for MLE:
$$
\\{(y\_i,\mathbf{x}'\_i)|i=1,...,N\\}
$$
$$
\mathbf{y}=\begin{bmatrix}
y\_1\\\\
\vdots\\\\
y\_N
\end{bmatrix}, \quad 
\mathbf{X}=\begin{bmatrix}
\mathbf{x}'\_1\\\\
\vdots\\\\
\mathbf{x}'\_N
\end{bmatrix}:N\times k
$$

SlideNext

# Bayes Estimator (continued)
- We consider the multiple regression model.
- As you will see, the discussion is generally the same as before,

SlideNextV
## Prior

$$
\begin{cases}
\sigma^{-2}  \sim  G(\underline{\delta},  \underline{\nu}),\\\\
\boldsymbol{\mu}|  \sigma^2  \sim  N(\underline{\boldsymbol{\mu}},  \underline{\boldsymbol{\Omega}}\sigma^2)
\end{cases}
$$
- For example, for a simple regression model, $\boldsymbol{\mu}=(a,b)'$ and prior hyperparameters are:
  + $\underline{\boldsymbol{\mu}}=(\underline{a},\underline{b})'$
  + $\underline{\boldsymbol{\Omega}}=
\begin{bmatrix}
\underline{\omega}\_{11}&\underline{\omega}\_{12}\\\\
\underline{\omega}\_{21}&\underline{\omega}\_{22}
\end{bmatrix}$

SlideNextV
## Posterior

$$
(\boldsymbol{\mu}',\sigma^{-2}|x_1,...,x_N)  \sim  \text{NG}(\overline{\boldsymbol{\mu}},\overline{\boldsymbol{\Omega}},\overline{\delta},\overline{\nu})
$$
- where:
$$
\begin{cases}
\overline{\boldsymbol{\Omega}}=(\underline{\boldsymbol{\Omega}}^{-1}+\mathbf{X'X})^{-1}\\\\
\overline{\boldsymbol{\mu}}=\overline{\boldsymbol{\Omega}}(\underline{\boldsymbol{\Omega}}^{-1}\underline{\boldsymbol{\mu}}+\mathbf{X'X}\hat{\boldsymbol{\mu}})\\\\
\overline{\nu}=\underline{\nu}+N\\\\
\overline{\delta}^{-1}=\frac{\underline{\nu}}{\underline{\nu}+N}\underline{\delta}+\frac{N-1}{\underline{\nu}+N}s^2+\frac{1}{\underline{\nu}+N}(\hat{\boldsymbol{\mu}}-\underline{\boldsymbol{\mu}})[\underline{\boldsymbol{\Omega}}+(\mathbf{X'X})^{-1}]^{-1}(\hat{\boldsymbol{\mu}}-\underline{\boldsymbol{\mu}})'
\end{cases}
$$

SlideNextV

## Some Notes
- Summation of the weights in $\overline{\boldsymbol{\mu}}$:
$$
\overline{\boldsymbol{\Omega}}\underline{\boldsymbol{\Omega}}^{-1}+\overline{\boldsymbol{\Omega}}\mathbf{X'X}=(\underline{\boldsymbol{\Omega}}^{-1}+\mathbf{X'X})^{-1}(\underline{\boldsymbol{\Omega}}^{-1}+\mathbf{X'X})=\mathbf{I}
$$
- Large number of observations:
$$N\rightarrow  \infty  \Rightarrow  \mathbf{X'X}  \rightarrow  [\infty]  \Rightarrow  \overline{\boldsymbol{\Omega}}\rightarrow  \mathbf{0}  \Rightarrow  \begin{cases}
\overline{\boldsymbol{\mu}}=\hat{\boldsymbol{\mu}}\\\\
\overline{\delta}=s^{-2}
\end{cases}
$$
- High prior uncertainty:
$$
\underline{\boldsymbol{\Omega}}  \rightarrow  [\infty]  \Rightarrow  \overline{\boldsymbol{\mu}}=\hat{\boldsymbol{\mu}}
$$
- Prior certainty:
$$
\underline{\boldsymbol{\Omega}}  \rightarrow  \mathbf{0}  \Rightarrow  \overline{\boldsymbol{\mu}}=\underline{\boldsymbol{\mu}}
$$

SlideNext

## Multivariate Regression (Unrestricted)
- The formal representation of the theory might need a multivariate statistical modelling.
- Consider the following set of distributions:
$$
\\{\frac{1}{\sqrt{(2\pi)^m|\boldsymbol{\Sigma}|}}e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})'}:~\mathbf{x},\boldsymbol{\mu}\in  \mathbb{R}^m,~\boldsymbol{\Sigma}\in  \mathbb{R}^{m\times  m}  \text{(p.d.)}\\}
$$
- This is a multivariate normal model.
- Like before, a generalization can be introduced as a Regression Model.
$$
\mathbf{y}=\mathbf{M}\mathbf{x}  +  \boldsymbol{\epsilon},\quad  \boldsymbol{\epsilon}\sim  iid.\mathbf{  N}(\mathbf{0},\boldsymbol{\Sigma}),
$$
- $\mathbf{y}:m\times  1$, $\mathbf{x}:k\times  1$, $\mathbf{M}:m\times  k$ and $\boldsymbol{\Sigma}:m\times  m$ (positive definite).
- The model is unrestricted; all equations have similar set of independent variables.

SlideNext

## MLE (Unrestricted)
- A more compact representation of the model is given in the following equation,
$$
\mathbf{y}=  (\mathbf{x}'  \otimes  \mathbf{I}_m)\boldsymbol{\mu}  +  \boldsymbol{\epsilon},  \quad  \boldsymbol{\mu}=  \operatorname{vec}  \mathbf{M}:mk\times  1
$$
- Assuming that there is a sample of size $N$ and observations are independent:
$$
L=(2\pi)^{-Nm/2}|\boldsymbol{\Sigma}|^{-N/2}\text{exp}(-\frac{1}{2}\underset{t=1}{\overset{N}{\sum}}\boldsymbol{\epsilon}'\_t\boldsymbol{\Sigma}^{-1}\boldsymbol{\epsilon}\_t),
$$

SlideNextV

## MLE Estimator
- Log-likelihood is maximized at:
$$
\begin{cases}
\hat{\mathbf{M}}_N=(\mathbf{Y'}\mathbf{X})(\mathbf{X'}\mathbf{X})^{-1},\\\\
\hat{\boldsymbol{\Sigma}}_N=\frac{1}{N}\mathbf{e'e}
\end{cases}
$$
in which $\mathbf{Y,e}:N\times  m$, $\mathbf{X}:N\times  k$, and
$$
\mathbf{e'e}=\underset{t=1}{\overset{N}{\sum}}\mathbf{e}_t\mathbf{e'}\_t,~~
\mathbf{Y'X}=\underset{t=1}{\overset{N}{\sum}}\mathbf{y}\_t\mathbf{x'}\_t,~~\mathbf{X'X}=\underset{t=1}{\overset{N}{\sum}}\mathbf{x}\_t\mathbf{x'}\_t,
$$
and,
$$
\mathbf{e}\_t=\mathbf{y}\_t-\hat{\mathbf{M}}\_N{}\mathbf{x}\_t.
$$

SlideNextV
## Asymptotic Distribution

$\hat{\mathbf{M}}\_N$ is actually the _Ordinary Least Square_ estimator, which under normality assumption is equal to Maximum Likelihood estimator.

$\hat{\boldsymbol{\mu}}\_N=\operatorname{vec}\hat{\mathbf{M}}\_N$ is asymptotically normally distributed, i.e.,
$$
\sqrt{N}(\hat{\boldsymbol{\mu}}\_N -  \boldsymbol{\mu})  \xrightarrow{d}  \mathbf{N}(\mathbf{0},\mathbf{Q}^{-1}  \otimes  \boldsymbol{\Sigma}),
$$
in which $\mathbf{Q}=E(\mathbf{x}\_t\mathbf{x}'\_t):k\times  k$ and is estimated consistently with
$$
\hat{\mathbf{Q}}=\frac{1}{N}\sum\_{t=1}^{N}\mathbf{x}\_t\mathbf{x}'\_t
$$

SlideNext

## Bayes Estimator (Continued)
- Recall the prior for variance in a single equation model:
$$ 
\sigma^{-2}  \sim  G(\underline{\delta},  \underline{\nu}) 
$$
- In a multivariate model, we are dealing with a matrix of covariances
  + We need a new type of prior
  + It must consider positive definiteness into account.
- Normal Wishart is a generalization of normal gamma.

SlideNextV

## Wishart Distribution
$$
f(\mathbf{X})=\frac{|\mathbf{X}|^{\frac{\nu-m-1}{2}}e^{\frac{-tr(\boldsymbol{\Delta}^{-1}\mathbf{X})}{2}}}{2^{\frac{\nu m}{2}}|\boldsymbol{\Delta}|^{\frac{\nu}{2}}\Gamma_{m}(\frac{\nu}{2})}, \; \boldsymbol{\Delta}\in\mathbb{R}^{m\times m} (p.d.), v\in \mathbb{R}, \nu>m-1
$$
- $\Gamma_{m}$ is multivariate gamma function.
- $tr$ is trace operator.
- $E(\mathbf{X})=\nu\boldsymbol{\Delta}$
- $var(\mathbf{X}\_{ij})=\nu(\delta\_{ij}^{2}+\delta\_{ii}\delta\_{jj})$
- notation: $\mathbf{W}(\boldsymbol{\Delta},\nu)$

SlideNextV

## A Prior
$$
\mathbf{y}=  (\mathbf{x}'  \otimes  \mathbf{I}_m)\boldsymbol{\mu}  +  \boldsymbol{\epsilon},\quad  \boldsymbol{\epsilon}\sim  iid.\mathbf{  N}(\mathbf{0},\boldsymbol{\Sigma}),
$$
- in which
$$
\begin{cases}
\boldsymbol{\Sigma}^{-1}\sim \mathbf{W}(\underline{\boldsymbol{\Delta}},\underline{v})\\\\
\boldsymbol{\mu}|\boldsymbol{\Sigma}  \sim  \mathbf{N}(\underline{\boldsymbol{\mu}},  \boldsymbol{\Sigma}\otimes\underline{\boldsymbol{\Omega}}) 
\end{cases}
$$

SlideNextV
## The Posterior
- Compared with the single equation case, $\overline{\mathbf{S}}$ is different:
$$
\overline{\boldsymbol{\Delta}}=\underline{\boldsymbol{\Delta}}+\hat{\boldsymbol{\Sigma}}_N+(\hat{\boldsymbol{\mu}}-\underline{\boldsymbol{\mu}})[\underline{\boldsymbol{\Omega}}+(\mathbf{X'X})^{-1}]^{-1}(\hat{\boldsymbol{\mu}}-\underline{\boldsymbol{\mu}})'
$$

SlideNext
## Noninformative Prior
- Priors can be categorized based on the variance of the distribution.
- An _informative prior_ has a relatively small variance.
  + It expresses specific information about a variable.
- A _noninformative_ prior has a very large variance.
  + It expresses vague or general information about a variable.
- There is another categorization too:
  + A prior distribution that integrates to 1 is a proper prior.
  + An _improper prior_ doesn't integrate to 1.
- noninformative priors tend to be improper in most models (Koop, 2003, p. 23).

SlideNextV

## An Example
- Normal model: $\underline{\beta}=0$ and $\underline{\omega} \rightarrow \infty$

SlideNext

## Multivariate Regression (Restricted)
- Equations have different set of regressors.
- A linear system of restrictions:
$$
\boldsymbol{\mu}=\mathbf{R}\boldsymbol{\mu}^r
$$
- $\mathbf{R}:mk\times q^*$ is a known matrix of rank $q^*$.
- $\boldsymbol{\mu}^r:q^* \times 1$ is a vector of unknown parameters. 
- Maximizing the log-likelihood function with respect to these constraints:
$$
\hat{\boldsymbol{\mu}}_N^r=[\mathbf{R'}(\mathbf{X'}\mathbf{X}\otimes \boldsymbol{\Sigma}^{-1})\mathbf{R}]^{-1}\mathbf{R'}(\mathbf{X'}\otimes \boldsymbol{\Sigma}^{-1})\operatorname{vec}(\mathbf{Y})
$$
- Known also as _Generalized Least Square_.
- It is asymptotically normally distributed:
$$
\sqrt{N}(\hat{\boldsymbol{\mu}}_N^r - \boldsymbol{\mu}) \overset{d}{\rightarrow} \mathbf{N}(\mathbf{0},\mathbf{S}),
$$
- in which $\mathbf{S}=[\mathbf{R'}(\mathbf{X'}\mathbf{X}\otimes \boldsymbol{\Sigma}^{-1})\mathbf{R}]^{-1}$

SlideNextV
## Estimated GLS
- The preceding estimator is not useful in practice
- It need a knowledge of $\boldsymbol{\Sigma}$.
- We need a consistent estimator of $\boldsymbol{\Sigma}$ to get the _Estimated GLS_.
- Which has the same asymptotic properties as the GLS:
- The unconstrained model can be used to estimate an consistent estimator for $\boldsymbol{\Sigma}$. 
- Alternatively, Least Square estimator for the constraint model can be used, i.e.,
$$
\tilde{\boldsymbol{\pi}}_N^r=[\mathbf{R'}(\mathbf{X'}\mathbf{X}\otimes \mathbf{I}_m)\mathbf{R}]^{-1}\mathbf{R'}(\mathbf{X'}\otimes \mathbf{I}_m)\operatorname{vec}(\mathbf{Y})
$$
- Which one has a better small sample properties?
- If one is confident about the validity of the restrictions (i.e., non-sample information exists), the second one is a better choice.

SlideNextV
## Bayes Estimator
- Assume an _Independent_ Normal-Wishart prior:
$$
\begin{cases}
\boldsymbol{\mu} \sim  \mathbf{N}(\underline{\boldsymbol{\mu}},  \underline{\boldsymbol{\Psi}})\\\\
\boldsymbol{\Sigma}^{-1}\sim \mathbf{W}(\underline{\mathbf{S}},\underline{v})
\end{cases}
$$
- It is slightly different from the one used in the unrestricted section. 
  + no conditional distribution.


SlideNextV
## Conditional Posterior
$$
\begin{cases}
\boldsymbol{\mu}|\boldsymbol{\Sigma},\mathbf{I} \sim  \mathbf{N}(\overline{\boldsymbol{\mu}},  \overline{\boldsymbol{\Psi}})\\\\
\boldsymbol{\Sigma}^{-1}|\boldsymbol{\mu},\mathbf{I}\sim \mathbf{W}(\overline{\boldsymbol{\Delta}},\overline{v})
\end{cases}
$$

$$
\begin{cases}
\overline{\boldsymbol{\Psi}}=(\underline{\boldsymbol{\Psi}}^{-1}+(\boldsymbol{\Sigma}^{-1}\otimes\mathbf{X'X}))^{-1}\\\\
\overline{\boldsymbol{\mu}}=\overline{\boldsymbol{\Psi}}(\underline{\boldsymbol{\Psi}}^{-1}\underline{\boldsymbol{\mu}}+(\boldsymbol{\Sigma}^{-1}\otimes\mathbf{X'X})\hat{\boldsymbol{\mu}})\\\\
\overline{\nu}=\underline{\nu}+N\\\\
\overline{\boldsymbol{\Delta}}^{-1}=\underline{\boldsymbol{\Delta}}+\hat{\boldsymbol{\Sigma}}_N+(\hat{\boldsymbol{\mu}}-\boldsymbol{\mu})\mathbf{X'X}(\hat{\boldsymbol{\mu}}-\boldsymbol{\mu})'
\end{cases}
$$
 
SlideNextV
 
## Gibbs Sampler
1. $i=0$
   - Calibrate $\boldsymbol{\mu}^{(i)}$ and $\boldsymbol{\Sigma}^{(i)}$.
   - Select Maximum number of iterations.
2. $i=i+1$
   - Draw $\boldsymbol{\mu}^{(i)}$ from $\boldsymbol{\mu}|\boldsymbol{\Sigma}^{(i-1)},\mathbf{I}$
   - Draw $\boldsymbol{\Sigma}^{(i)}$ from $\boldsymbol{\Sigma}|\boldsymbol{\mu}^{(i-1)},\mathbf{I}$
3. Save the results.
4. If maximum iteration is not reached, go to step 2.
5. Keep a  _subset_ of the results.

SlideNext

## Dynamic Regression Model
- Recall the simple linear regression model:
$$
y\_t=a+b x\_t+\epsilon\_t,  \;  \epsilon\_t\sim  N(0,\sigma^2)
$$ 
- Now, assume that the lag of independent variable is the regressor:
$$
y\_t=a+b y\_{t-1}+\epsilon\_t,
$$
- This affects joint probability distribution (or likelihood function) calculations.
  + Observations are not independent anymore.

SlideNextV

## Conditional Likelihood Function
- Calculating the exact likelihood function is possible.
- However, assuming that $y_0$ is given is a common practice.
$$
f(y\_1,y\_2,\ldots,y\_T|y\_0)=f(y\_1|y\_0)\times f(y\_2|y\_1)\ldots f(y\_N|y\_{N-1})
$$ 
- The rest is similar to the previous discussion.

SlideNext

## Stationary VAR
- A (Stationary) $\text{VAR}(p)$ model is represented by,
$$
\mathbf{y}\_t=\underset{i=1}{\overset{N}{\sum}}\boldsymbol{\Phi}\_i \mathbf{y}\_{t-i}+\boldsymbol{\Pi}\mathbf{x}\_t + \boldsymbol{\epsilon}\_t,\quad \boldsymbol{\epsilon}\_t\sim iid.\mathbf{N}(\mathbf{0},\boldsymbol{\Sigma}).
$$
- $\mathbf{y}_t:m\times 1$: the vector of dependent variables;
- $\mathbf{y}_{t-i}:m\times 1$: the $i$-th lag of dependent variables;
- $\mathbf{x}_t:k\times 1$: the vector of exogenous variables;
- $\boldsymbol{\Pi}:m\times k$,  and $\boldsymbol{\Phi}_i,\boldsymbol{\Sigma}:m\times m$: the coefficients of the model ($\boldsymbol{\Sigma}$ is positive definite);
- $\boldsymbol{\epsilon}_t$ is the vector of the disturbances.

SlideNextV

## Stationarity
- All the roots of the following equation lie outside the unit circle:
$$
|\mathbf{I}-\boldsymbol{\Phi}_1z-\ldots-\boldsymbol{\Phi}_pz^p|=0.
$$ 
 
SlideNextV
 
## Another Representation
 
- We can easily change the representation to a _SUR_ model representation:
 $$
\mathbf{y}_t=\boldsymbol{\Gamma}\mathbf{z}_t + \boldsymbol{\epsilon}_t,
$$
- in which
$$
\boldsymbol{\Gamma}=[\boldsymbol{\Phi_1} \ldots \boldsymbol{\Phi_p}, \boldsymbol{\Pi}]:m\times (mp+k)
$$
$$
\mathbf{z}'\_t=[\mathbf{y}'\_{t-1}\ldots\mathbf{y}'\_{t-p},\mathbf{x}'\_t]:1\times (mp+k).
$$

SlideNextV

## MA Representation
- Due to the dynamic nature of the model, a part of the literature is about another representation of this model
- Moving average (MA) representation. 
- This is derived by using _lag_ operator, or recursive substitutions:
$$
\mathbf{y}\_t=\underset{i=0}{\overset{\infty}{\sum}}\boldsymbol{\Psi}\_i (\boldsymbol{\Pi}\mathbf{z}\_{t-i} +\boldsymbol{\epsilon}\_{t-i}),
$$
- in which $\boldsymbol{\Psi}_0=\mathbf{I}$ and,
$$
\boldsymbol{\Psi}\_i=\boldsymbol{\Psi}\_{i-1}\boldsymbol{\Phi}\_1 + \ldots + \boldsymbol{\Psi}\_{i-p}\boldsymbol{\Phi}\_p
$$
 - with $\boldsymbol{\Psi}\_{i}=\mathbf{0}$  for $i<0$. 

SlideNext

## Conditional Likelihood Function
- Assuming that there are $T$ observations, recall the likelihood function of non dynamic model:
$$
L=(2\pi)^{-Tm/2}|\boldsymbol{\Sigma}|^{-T/2}\text{exp}(-\frac{1}{2}\sum_{t=1}^{T}\boldsymbol{\epsilon}'_t\boldsymbol{\Sigma}^{-1}\boldsymbol{\epsilon}_t).
$$

- This function is calculated based on the assumption that the observations are independent from each other. 
- This is not the case in a VAR model, because each observation is related to previous observations. 
- The preceding formula is actually the _Conditional Likelihood_ function, in which the value of $\mathbf{y}\_0$, $\mathbf{y}\_{-1},...,\mathbf{y}\_{-p+1}$ are regarded deterministic.

SlideNext

## Bayesian Analysis
- Recall the Normal-Wishart prior:
$$
\begin{cases}
\boldsymbol{\mu} \sim  \mathbf{N}(\underline{\boldsymbol{\mu}},  \underline{\boldsymbol{\Psi}})\\\\
\boldsymbol{\Sigma}^{-1}\sim \mathbf{W}(\underline{\boldsymbol{\Delta}},\underline{v})
\end{cases}
$$
- recall the $i$-th equation in a VAR model:
- $y\_{it}=c\_i+\phi\_{i1}^1  y\_{1,t-1}+\phi\_{i2}^1  y\_{2,t-1}+…+\phi\_{in}^1 y\_{n,t-1}+$

$\quad\phi\_{i1}^2 y\_{1,t-2}+\phi\_{i2}^2  y\_{2,t-2}+…+\phi\_{in}^2  y\_{n,t-2}+…+$

$\quad\phi\_{i1}^p y\_{1,t-p}+\phi\_{i2}^p y\_{2,t-p}+…+\phi\_{in}^p y\_{n,t-p}+\epsilon\_{it}$

- The parameters are a part of $\boldsymbol{\mu}$.

SlideNextV
## Non-Data-Based Information
- If the subject of the VAR model is macroeconomics, we might have some _non-data-based_ information about the variables!
  + Many of them might be I(1).
  + $\Delta y\_{it}=c\_i+\epsilon\_{it}$

SlideNextV
## The Minnesota Prior
- It assumes that prior covariance matrix is diagonal.
- Probably, the coefficient of the lag of the dependent variable is $1$;
$$
\phi\_{ii}^1  \sim N(1,\gamma^2), \quad 
$$
  + $\gamma = 0.2 \Leftrightarrow P(0.6<\phi\_{ii}^1<1.4)=0.95$
- Probably, as lag length increases, coeffcients are increasingly shrunk towards zero.
$$
\phi\_{ii}^s  ~ \sim N(0,(\frac{\gamma}{s})^2),\quad  s>1
$$
 
 
SlideNextV
## The Minnesota Prior (continued)
- Probably, the lags of other variables are zero and this guess is more valid as lag length increases.
$$
\phi\_{ij}^s  ~ \sim N(0,(\frac{\gamma}{s})^2\times w^2 \times (\frac{\hat{\tau}\_i}{\hat{\tau}\_j})^2),\quad  i\ne j
$$
- $\hat{\tau}\_i$ is the estimated standard deviation of the residuals in an $AR(p)$ model.
- $\frac{\hat{\tau}\_i}{\hat{\tau}\_j}$ controls difference in measurement units between $i$ and $j$-th variables.
- We might be are more certain that the lags of other variables are zero, than the lags of the dependent variable. 
  + $w=0.5$

 
SlideNext
The End


</section> 

