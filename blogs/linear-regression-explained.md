# Linear Regression, Explained Like You're Pricing Insurance for the First Time

Imagine you just got hired at a health insurance company, and your very first task is this: given a new customer's age, sex, BMI, number of children, whether they smoke, and where they live, figure out how much to charge them. No formula exists yet. Just a spreadsheet with 1,338 past customers and what they were actually charged.

This is exactly the problem Linear Regression was built to solve, and it is exactly the problem this note walks through, start to finish, using a real Medical Cost dataset with those exact six features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) predicting one target: `charges`.

By the end, you will understand not just how to run Linear Regression, but exactly what is happening mathematically underneath it, including where its famous formula comes from and how to judge whether your predictions are actually any good.

<div class="blog-image">
  <img
    src="../assets/images/blogs/linear-regression/best_fit_line.png"
    alt="Best fit line intuition"
    style="max-width: 100%; height: auto;"
  />
  <p><em>Example of candidate lines vs the least-squares best-fit line.</em></p>
</div>

## What Is Linear Regression, Really?

Linear Regression is a supervised learning algorithm used whenever the thing you are trying to predict is a continuous real number, like a price, a temperature, or, in this case, an insurance charge. It works by finding the best fit line, or in higher dimensions, the best fit flat surface, that describes the relationship between your independent variables (age, BMI, and so on) and your dependent variable (charges).

It does this using a principle called Ordinary Least Squares, usually shortened to OLS. The goal of OLS is simple to state: find the line that minimizes the total squared difference between what the model predicts and what actually happened in the data. Squaring the differences matters here, since it treats being 100 dollars too high the same as being 100 dollars too low, and it punishes big mistakes disproportionately more than small ones.

## Setting Up the Problem: The Hypothesis Function

In Linear Regression, each row of your data is called a training example. If you have one independent variable `x` and one dependent variable `y`, a single training example is written as the pair `(xi, yi)`, where the subscript `i` just means "the i-th row." If you have `m` total rows, then `i` runs from 1 up to `m`.

The whole goal of supervised learning is to learn a function, called the hypothesis function, written `h(x)`, that takes an input `x` and estimates the corresponding `y`.

### Simple Linear Regression: One Feature

If you only had one feature, say, just BMI, predicting charges, the hypothesis function would look like this:

```
h(xi) = θ0 + θ1 * xi
```

This is just the equation of a straight line. `θ0` is the intercept, where the line crosses the y-axis, and `θ1` is the slope, how steeply charges increase as BMI increases. `θ0` and `θ1` are called the parameters of the hypothesis, and the entire job of "training" a Linear Regression model is really just finding the specific numeric values of these parameters that fit the data best.

### Multiple Linear Regression: Many Features

Our actual dataset has six independent variables, not one, so we need Multiple Linear Regression instead. Now each training example has several feature values, written `xi1, xi2, ..., xin`, where `j` runs from 1 to `n`, the number of features. The hypothesis function extends naturally:

```
h(xi) = θ0 + θ1*xi1 + θ2*xi2 + ... + θn*xin
```

For our dataset specifically, with `age`, `sex`, `bmi`, `children`, `smoker`, and `region` as the six features, this becomes:

```
h(xi) = θ0 + θ1*age + θ2*sex + θ3*bmi + θ4*children + θ5*smoker + θ6*region
```

<div class="blog-image">
  <img
    src="../assets/images/blogs/linear-regression/hyperplane.png"
    alt="Linear regression hyperplane intuition"
    style="max-width: 100%; height: auto;"
  />
  <p><em>Generalizing from a line (2D) to a hyperplane in higher dimensions.</em></p>
</div>

Plugging in the very first row of real data, where a 19-year-old female, BMI 27.9, no children, a smoker, from the southwest region, was actually charged 16,884.92:

```
h(x1) = θ0 + θ1*19 + θ2*female + θ3*27.900 + θ4*0 + θ5*yes + θ6*southwest
y1 = 16884.92400
```

The model's whole job is to find values of `θ0` through `θ6` so that `h(x1)` comes out as close as possible to 16,884.92, and simultaneously as close as possible to every other `yi` in the dataset too, all at once. That balancing act across every single row is exactly what the cost function, covered shortly, measures.

## Rewriting Everything as Matrices

Writing out this hypothesis function separately for all 1,338 rows would be unbearable. So instead, everything gets packed into matrices and vectors, which is really just a compact bookkeeping trick, not a new mathematical idea.

Every individual row's features get stacked into one big input matrix `X`, of size `(m, n)`, meaning `m` rows and `n` columns, one row per training example and one column per feature.

The parameters `θ0` through `θn` get stacked into a single column vector `θ`, of size `(n+1, 1)`. The target values `y1` through `ym` get stacked into their own column vector `y`, of size `(m, 1)`.

With everything in this form, the entire hypothesis function for every single row, all at once, collapses into one clean line:

```
h(x) = Xθ
```

That is the entire point of the matrix formulation: instead of writing the hypothesis equation 1,338 separate times, one matrix multiplication does all of it simultaneously. As a quick sanity check on why the shapes work out: matrix multiplication only works when the number of columns in the first matrix matches the number of rows in the second. `X` is `(m, n+1)` and `θ` is `(n+1, 1)`, so multiplying them is valid and produces a result of shape `(m, 1)`, exactly matching `y`. The shapes lining up like this is a good way to sanity check that a matrix equation makes sense before trusting it.

## The Cost Function: Measuring How Wrong the Model Is

A cost function measures how much error the model has, in terms of how far off its predictions are from reality. The specific cost function Linear Regression uses is:

```
J(θ) = (1/m) * Σ (h(xi) - yi)²
```

In plain words: for every row, take the model's prediction, subtract the actual value, square that difference, then average all those squared differences across every row. This single number, `J(θ)`, tells you exactly how good or bad a particular choice of `θ` is. A perfect model would have `J(θ) = 0`. Training the model means searching for the specific `θ` values that make `J(θ)` as small as possible.

You will also often see this same quantity called the RSS, short for Residual Sum of Squares, usually referring to the sum of squared errors without dividing by `m`. `J(θ)` here is just the RSS averaged across all rows, and some textbooks or tutorials use `θ` for the parameter vector while others use `β` (beta) instead. Both refer to the exact same thing: the vector of parameters the model is trying to solve for.

To make the matrix version of this cost function work out cleanly, one small trick gets added: an extra column of all 1s gets bolted onto the front of `X`, corresponding to `θ0`, the intercept term. This turns `X` into a matrix of shape `(m, n+1)` instead of `(m, n)`. With that adjustment, the cost function can be written entirely in matrix form:

<div class="blog-image">
  <img
    src="../assets/images/blogs/linear-regression/residuals.png"
    alt="Residuals (errors) for the fitted line"
    style="max-width: 100%; height: auto;"
  />
  <p><em>Vertical residuals (errors) show what the cost function squares and sums.</em></p>
</div>

```
J(θ) = (1/m) * (Xθ - y)ᵀ * (Xθ - y)
```

This is exactly the same calculation as before, just expressed as one matrix expression instead of a summation. `(Xθ - y)` is a column vector of every row's individual error. Multiplying it by its own transpose is a compact way of squaring every entry and adding them all together in a single step.

## The Normal Equation: Solving for the Best θ Directly

Here is where Linear Regression gets genuinely elegant. Rather than guessing and slowly adjusting `θ` bit by bit, it turns out you can solve directly, using calculus, for the exact `θ` that minimizes `J(θ)` in one shot. This direct solution is called the Normal Equation.

The idea from calculus is simple: at the lowest point of a curve, its slope is exactly zero. So, to find the `θ` that minimizes the cost function, take the derivative of `J(θ)` with respect to `θ`, and set it equal to zero.

```
∂J(θ)/∂θ = 0
```

Here is the derivation, one step at a time, dropping the constant `1/m` upfront since it does not affect where the minimum sits.

Start by expanding `J(θ) = (Xθ - y)ᵀ(Xθ - y)` using ordinary algebra:

```
J(θ) = θᵀXᵀXθ - 2θᵀXᵀy + yᵀy
```

Now take the derivative of each piece with respect to `θ`, using two basic calculus facts: the derivative of `θᵀAθ` with respect to `θ` is `2Aθ` when `A` is symmetric (and `XᵀX` always is), and the derivative of a constant, like `yᵀy`, is simply 0.

```
∂J(θ)/∂θ = 2XᵀXθ - 2Xᵀy + 0
```

Setting this equal to zero, exactly as the calculus rule for a minimum requires:

```
0 = 2XᵀXθ - 2Xᵀy
XᵀXθ = Xᵀy
```

And finally, solving for `θ` by multiplying both sides by the inverse of `XᵀX`:

```
θ = (XᵀX)⁻¹ Xᵀy
```

This is the Normal Equation.

## Fitting a Quick Visual: Charges vs BMI

Before judging any model numerically, it always helps to see the relationship visually first. Plotting just BMI against charges, ignoring the other five features for a moment, and fitting a regression line through it:

```python
sns.lmplot(x='bmi', y='charges', data=df, aspect=2, height=6)
plt.xlabel('Body Mass Index (kg/m²): Independent variable')
plt.ylabel('Insurance Charges: Dependent variable')
plt.title('Charges vs BMI')
```

## How Do You Know If the Model Is Any Good? Evaluation Metrics

### Mean Absolute Error (MAE)

```
MAE = (1/N) * Σ |yi - ŷi|
```

### Mean Squared Error (MSE)

```
MSE = (1/N) * Σ (yi - ŷi)²
```

### Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √[(1/N) * Σ (yi - ŷi)²]
```

<div class="blog-image">
  <img
    src="../assets/images/blogs/linear-regression/metrics_outlier_sensitivity.png"
    alt="Outlier sensitivity of error metrics"
    style="max-width: 100%; height: auto;"
  />
  <p><em>MSE/RMSE vs MAE: squared errors amplify outliers.</em></p>
</div>

### R² Score (Coefficient of Determination)

```
R² = 1 - [Σ(yi - ŷi)² / Σ(yi - ȳ)²]
```

## Bringing It All Together

That is the complete arc of classical Linear Regression.

## Building Linear Regression From Scratch in Python, No Library Doing the Math for You

<div class="blog-image">
  <img
    src="../assets/images/blogs/linear-regression/gd_vs_closedform.png"
    alt="Gradient descent vs closed form"
    style="max-width: 100%; height: auto;"
  />
  <p><em>Gradient descent (iterative) compared to the closed-form (Normal Equation) approach.</em></p>
</div>

```python
import numpy as np


class LinearRegressionClosed:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        ones_column = np.ones((X.shape[0], 1))
        Xb = np.concatenate([ones_column, X], axis=1)

        beta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y)

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self, X):
        X = np.array(X)
        return X.dot(self.coef_) + self.intercept_
```

## Testing It Against a Real Dataset

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

X, y = fetch_california_housing(return_X_y=True)

reg = LinearRegressionClosed()
reg.fit(X, y)
y_pred = reg.predict(X)
print("Our from-scratch R2 score:", r2_score(y, y_pred))

reg_sk = LinearRegression()
reg_sk.fit(X, y)
y_pred_sk = reg_sk.predict(X)
print("scikit-learn R2 score:", r2_score(y, y_pred_sk))
```

