---
layout: presentation  
htmllang: en  
heading: Estimation Framework  
---
  

<section data-markdown data-separator="^SlideNext$" data-separator-vertical="^SlideNextV$">  
  

# Estimation Framework  
  

Ramin Mojab, 2018  
  

SlideNext
  

## Where Do Parameters Come From?  

- A **Statistical Model** is a set of _statistical assumptions_ about a population.  
- It is a formal representation of a theory and an essential part of any statistical inference. It consists of:  
1. a set of possible observations (a sample space and a (non-empty) set of outcomes);  
1. a **set of** probability distributions.  
  

SlideNextV
  

## Where Do Parameters Come From? (continued)  
  

- A distribution almost always contains parameters.  
  + The set of probability distributions of the model can be built using such a set; i.e., a probability distribution is assignsed to each parameter point.  
- As a result, a statistical model consists of unknown parameters, i.e., **Statistical Parameters**.  
  

SlideNextV
  

## Descriptive Statistics  

- This consists of providing visual or quantitative summaries about a set of data (or observations).  
- We might use simple plots or calculate percentages, means, variances, etc.  
- We generally provide information about two types of measures: central tendency and dispersion.  
- Where does the sample come from? What is the population? How is data distributed there? What happens to the mean or variance if we double the size of the sample?  
  + We do not deal with such questions in this subject  
Assuming that the data is just a sample of a larger population, **inferential statistics** can help us infer properties of the population.  
  

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
  

## How to approximate the Unknown Parameters?  

- What do we mean by _unknown_?!  
  + We don't know the exact value; however,  
  + The model might contain additional information about them (i.e., prior distributions).  
- More information comes form the _events_ or data.  
  + _Measured Empirical Data_, _sample data_ or _observations_ is any measurable subset of the sample space of the statistical model.  
- An **Estimator** approximates the statistical parameter using the measured empirical data.  
- It is a _sample statistic_.  
  + which means it can be computed by using the observations.  
  

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
  
- data (i.e., a set such as $\\{x_i\\}$ for $i=1,...,N$) comes from the events. Prior distributions might exist, too.  
- The question is about estimation, i.e., choosing a subset of the set of the distributions.  
  

SlideNextV
  

## Normal Distribution  
  

* Probably the most important probability distributions.  
- why?!  
- Central Limit Theorem  
  + roughly speaking, the average of independent random variables will converge to a normal distribution, as the number of observations increases  
* Many things follow its rules!  
- Again, due to the central limit theorem  
- If an event is the sum of other (independent) events, it is distributed normally.  
* It is also called _Gaussian_ distribution.  
- It was discovered by Carl Friedrich Gauss (1777-1855).  
* A version of it is called _Normal Standard_ or _z distribution_.  
  
  
  

SlideNextV
  

## Probability Density Function  
  

\\[f(x)  =  \frac{1}{\sqrt{2\pi}\sigma}  \cdot  e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}  \\]  
  

- \\(x,\mu,\sigma  \in  \mathbb{R},  \sigma> 0\\)  
  
- mean: \\(\mu\\)  
- variance: \\(\sigma^2\\)  
- notation: \\(  \mathcal{N}(\mu,\sigma^2)  \\)  
  
- zero variance?! not a traditional function; It is zero everywhere except \\(x=\mu\\) and its integral equals one!  
  

SlideNextV
  

## Bell Curve  

![Bell Curve Image](../../../assets/svgs/distributions/bellcurve.svg)  
  

SlideNextV
  

## Density?!  
  

- Well, we are interested in areas under portions of a normal distribution.  
  + why?! For example, we need them for hypothesis testing  
  + They Can be computed by using integral calculus. Of course, there is no exact _closed form_ expression for the integral over an arbitrary range.  
  + Two general useful rules:  
* 68% of the area is within one standard deviation of the mean.  
* 95% of the area is within 1.96 standard deviations of the mean.  
  

SlideNextV
  

![Normal 1](../../../assets/svgs/distributions/normal1.svg)  
  

SlideNextV
  

![Normal 2](../../../assets/svgs/distributions/normal2.svg)  
  

SlideNextV
  

![Normal 3](../../../assets/svgs/distributions/normal3.svg)  
  

SlideNextV
  

## Cumulative Density Functions  
  

\\[F(x)=\frac{1}{2}  [1+\operatorname{erf}(\frac{x-\mu}{\sigma\sqrt{2}})]\\]  
  

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
  

![Inverse Gamma 1](../../../assets/svgs/distributions/gamma1.svg)  
  

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
\underline{\mu}\in  \mathbb{R},~\underline{\omega},\underline{\alpha},\underline{\beta}\in  \mathbb{R}\_{+}.  
$$  
- There are four _prior hyperparameters_.  
  

SlideNextV
  

## The Prior  

$$  
\sigma^{-2}  \sim  G(\underline{\alpha},  \underline{\beta}),  \quad\mu|  \sigma^2  \sim  N(\underline{\mu},  \underline{\omega}\sigma^2)  
$$  

- The first distribution is unconditional and therefore calibration is straightforward.  
- The other is conditional, which means, your belief about $\mu$ is not independent from the value of $\sigma^2$.  
  + Can't we use the more convenient unconditional distribution in this case?  
  + We can, however, there will be no analytical solution, just numerical ones. (This is where Gibbs or Metropolis-Hastings samplers are used).
  + In other words, there is a trade-off between the simplicity of the prior and the simplicity of the calculations.  
- Another point: Can we use a normal distribution for the variance?  
  + no, the variance cannot be negative.  
  

SlideNextV
  

## Posterior  

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
  

## Posterior 

$$  
P(\mu,\sigma^2|x_1,...,x_N)  \propto  f(x_1,...,x_N|\mu,\sigma^2)\times  NG(\underline{\mu},\underline{\omega},\underline{\alpha},\underline{\beta})  
$$  
We can show that,  
$$  
(\mu,\sigma^2|x_1,...,x_N)  \sim  NG(\overline{\mu},\overline{\omega},\overline{\alpha},\overline{\beta})  
$$  
where,  
$$  
\begin{cases}  
\overline{\omega}=(\underline{\omega}^{-1}+N)^{-1}\\\\  
\overline{\mu}=\overline{\omega}(\underline{\omega}^{-1}\underline{\mu}+N\hat{\mu})\\\\  
\overline{\beta}=\underline{\beta}+N\\\\  
\overline{\alpha}=\frac{\underline{\beta}}{\underline{\beta}+N}\underline{\alpha}+\frac{N-1}{\underline{\beta}+N}s^2+\frac{1}{\underline{\beta}+N}(\hat{\mu}-\underline{\mu})^{2}[\underline{\omega}+N^{-1}]^{-1}  
\end{cases}  
$$

SlideNextV

# Some Notes  

- The sum of weights in $\overline{\beta}$ equals $1$.  
- Large number of observations:  
$$N\rightarrow  \infty  \Rightarrow  \overline{\omega}\rightarrow  0  \Rightarrow  \begin{cases}  
\overline{\mu}=\hat{\mu}\\\\  
\overline{\alpha}=s^2  
\end{cases}  
$$  
- High prior uncertainty: $\underline{\omega}  \rightarrow  \infty  \Rightarrow  \overline{\mu}=\hat{\mu}$  
- Prior certainty: $\underline{\omega}  \rightarrow  0  \Rightarrow  \overline{\mu}=\underline{\mu}$  
- Equal weights: $\underline{\omega}=\frac{1}{N}  \Rightarrow  \overline{\mu}=\frac{\underline{\mu}+\hat{\mu}}{2}$
- 
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
 
# Regression Model

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
  

# Theoretical Model  

- Where does $y, x, a, b$, and the linear relationship come from?  
- Let $y$ be a person’s wage (measured in dollars per hour), and $x$ be her observed years of education.  
- Assume that there is a theory which points to a mathematical equation that describes a linear relationship between these two variables;  
$$  
y=a+b x,  \;  a,b\in  \mathbb{R}  
$$  
- Note that the theory might restrict the parameters; e.g., $b>0$.  
  

SlideNextV
  

# Theoretical Model vs. Regression Model  

- _Formal_ theoretical modeling is a good starting point for empirical analysis, but it is not essential.  
  + It might rely entirely on intuition or common sense.  
- The Regression model is a generalization of the theoretical model:  
  + The form of the functions must be specified in a regression model.  
  + The choice of the variables in a regression model is determined by data considerations, as well as theory.  
  + In a regression model, we should deal with the _disturbance term_.  
  

SlideNext
  

# MLE (continued)  

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

# MLE (Observed Hessian)  
  

$$  
\text{Hessian}:  \mathbf{H}=  
\begin{bmatrix}  
\frac{-N}{\sigma^{2}}  & -\frac{1}{\sigma^4}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i  \\\\  
-\frac{1}{\sigma^4}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i&\frac{N}{2\sigma^4}-\frac{1}{\sigma^6}\underset{i=1}{\overset{N}{\sum}}\hat{e}_i^2  
\end{bmatrix}  
$$  
  

- Use FOCs to show this matrix is diagonal and negative definite at the solution.  
  

SlideNextV
  

# The MLE Estimator

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

# Multiple Regression  
  

- The generalization for manipulating more variables from theory is straight-forward.  
- ...  
  

SlideNext
  

# Bayes Estimator (continued)

- We consider the multiple regression model.  
- As you will see, the discussion is generally the same as before,  
  

SlideNextV

# Prior  
  

$$
\begin{cases}  
\sigma^{-2}  \sim  G(\underline{\alpha},  \underline{\beta}),\\\\  
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

# Posterior  
  

$$  
(\boldsymbol{\mu}',\sigma^2|x_1,...,x_N)  \sim  \text{NG}(\overline{\boldsymbol{\mu}},\overline{\boldsymbol{\Omega}},\overline{\alpha},\overline{\beta})  
$$  

- where:  
$$  
\begin{cases}  
\overline{\boldsymbol{\Omega}}=(\underline{\boldsymbol{\Omega}}^{-1}+\mathbf{X'X})^{-1}\\\\  
\overline{\boldsymbol{\mu}}=\overline{\boldsymbol{\Omega}}(\underline{\boldsymbol{\Omega}}^{-1}\underline{\boldsymbol{\mu}}+\mathbf{X'X}\hat{\boldsymbol{\mu}})\\\\  
\overline{\beta}=\underline{\beta}+N\\\\  
\overline{\alpha}=\frac{\underline{\beta}}{\underline{\beta}+N}\underline{\alpha}+\frac{N-1}{\underline{\beta}+N}s^2+\frac{1}{\underline{\beta}+N}(\hat{\boldsymbol{\mu}}-\underline{\boldsymbol{\mu}})[\underline{\boldsymbol{\Omega}}+(\mathbf{X'X})^{-1}]^{-1}(\hat{\boldsymbol{\mu}}-\underline{\boldsymbol{\mu}})'  
\end{cases}  
$$  
  

SlideNextV
  

# Some Notes  

- Summation of the weights in $\overline{\boldsymbol{\mu}}$:  
$$  
\overline{\boldsymbol{\Omega}}\underline{\boldsymbol{\Omega}}^{-1}+\overline{\boldsymbol{\Omega}}\mathbf{X'X}=(\underline{\boldsymbol{\Omega}}^{-1}+\mathbf{X'X})^{-1}(\underline{\boldsymbol{\Omega}}^{-1}+\mathbf{X'X})=\mathbf{I}  
$$  
- Large number of observations:  
$$N\rightarrow  \infty  \Rightarrow  \mathbf{X'X}  \rightarrow  [\infty]  \Rightarrow  \overline{\boldsymbol{\Omega}}\rightarrow  \mathbf{0}  \Rightarrow  \begin{cases}  
\overline{\boldsymbol{\mu}}=\hat{\boldsymbol{\mu}}\\\\  
\overline{\alpha}=s^2  
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
  

# Multivariate Regression  

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
  

SlideNext
  

# MLE (continued)  

- A more compact representation of the model is given in the following equation,  
$$  
\mathbf{y}=  (\mathbf{x}'  \otimes  \mathbf{I}_m)\boldsymbol{\mu}  +  \boldsymbol{\epsilon},  \quad  \boldsymbol{\mu}=  \operatorname{vec}  \mathbf{M}:mk\times  1  
$$  
- Assuming that there is a sample of size $N$ and observations are independent:  
$$  
L=(2\pi)^{-Nm/2}|\boldsymbol{\Sigma}|^{-N/2}\text{exp}(-\frac{1}{2}\underset{t=1}{\overset{N}{\sum}}\boldsymbol{\epsilon}'\_t\boldsymbol{\Sigma}^{-1}\boldsymbol{\epsilon}\_t),  
$$  
  

SlideNextV

# MLE Estimator  

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

# Asymptotic Distribution  
  

$\hat{\mathbf{M}}\_N$ is actually the _Ordinary Least Square_ estimator, which under normality assumption is equal to Maximum Likelihood estimator.  
  

$\hat{\boldsymbol{\mu}}\_N=\operatorname{vec}\hat{\mathbf{M}}\_N$ is asymptotically normally distributed, i.e.,  
$$  
\sqrt{N}(\hat{\boldsymbol{\mu}}\_N -  \boldsymbol{\mu})  \xrightarrow{d}  \mathbf{N}(\mathbf{0},\mathbf{Q}^{-1}  \otimes  \boldsymbol{\Sigma}),  
$$  
in which $\mathbf{Q}=E(\mathbf{x}\_t\mathbf{x}'\_t):k\times  k$ and is estimated consistently with  
$$  
\hat{\mathbf{Q}}=\frac{1}{N}\sum\_{t=1}^{N}\mathbf{x}\_t\mathbf{x}'\_t  
$$  
  

SlideNextV
  

## Adjusting Degrees of Freedom  

- When $N$ is relatively small, an adjustment for degrees of freedom leads to an unbiased estimator of the residuals covariance matrix. i.e.,  
$$  
\tilde{\boldsymbol{\Sigma}}_N=\frac{\mathbf{e'e}}{N-k}  
$$  
- Note that there is $k$ parameters in each of the equations.  
  

SlideNextV
  

## T-Test  

- Calculating t-statistics is straightforward  
- The choice of its degrees of freedom might be controversial.  
- Since $mN$ observations is used to estimate $mk$ parameters, $m(N-k)$ is a choice.  
  

SlideNextV
  

## Prediction Error  

- Assuming that $\mathbf{x}\_t=\mathbf{x}\_t^0$, the prediction is:  
$$  
\mathbf{y}\_t^0=  (\mathbf{x}\_t^{'0}  \otimes  \mathbf{I}\_m)\hat{\boldsymbol{\mu}}\_T,  
$$  
- Prediction error is calculated by:  
$$  
\mathbf{e}\_t^0=\mathbf{y}\_t-\mathbf{y}\_t^0=  (\mathbf{x}\_t^{'0}  \otimes  \mathbf{I}\_m)(\boldsymbol{\pi}-\hat{\boldsymbol{\mu}}\_T)  +  \boldsymbol{\epsilon}\_t  
$$  
- The two terms in the preceding summation are independent, and therefore,  
$$  
\operatorname{var}(\mathbf{e}\_t^0)  =  (\mathbf{x}\_t^{'0}  \otimes  \mathbf{I}\_m)  
(\mathbf{Q}^{-1}  \otimes  \boldsymbol{\Sigma})  
(\mathbf{x}\_t^{0}  \otimes  \mathbf{I}\_m)+\boldsymbol{\Sigma}  
$$  
  

SlideNext
  

# Bayes Estimator (Continued)  
- Recall the prior for variance in a single equation model:
$$ 
\sigma^{-2}  \sim  G(\underline{\alpha},  \underline{\beta}) 
$$
- In a multivariate model, we are dealing with a matrix of covariances  
  + We need a new type of prior  
  + It must consider positive definiteness into account.  
- Normal Wishart is a generalization of normal gamma.  
  
SlideNextV
  
# Wishart Distribution
- ...

SlideNextV

# Prior
$$  
\mathbf{y}=  (\mathbf{x}'  \otimes  \mathbf{I}_m)\boldsymbol{\mu}  +  \boldsymbol{\epsilon},\quad  \boldsymbol{\epsilon}\sim  iid.\mathbf{  N}(\mathbf{0},\boldsymbol{\Sigma}),  
$$
- in which
$$
\begin{cases}
\boldsymbol{\Sigma}^{-1}\sim \mathbf{W}(\underline{\mathbf{S}},\underline{v})\\\\
\boldsymbol{\mu}|\boldsymbol{\Sigma}  \sim  \mathbf{N}(\underline{\boldsymbol{\mu}},  \underline{\boldsymbol{\Omega}}\sigma^2) 
\end{cases}
$$




SlideNext
The End  
  
  

</section> 