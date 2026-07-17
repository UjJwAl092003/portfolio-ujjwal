# Linear Regression, Explained Like You're Pricing Insurance for the First Time

Imagine you just got hired at a health insurance company, and your very first task is this: given a new customer's age, sex, BMI, number of children, whether they smoke, and where they live, figure out how much to charge them. No formula exists yet. Just a spreadsheet with 1,338 past customers and what they were actually charged.

This is exactly the problem Linear Regression was built to solve, and it is exactly the problem this note walks through, start to finish, using a real Medical Cost dataset with those exact six features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) predicting one target: `charges`.

By the end, you will understand not just how to run Linear Regression, but exactly what is happening mathematically underneath it, including where its famous formula comes from and how to judge whether your predictions are actually any good.

![Which line best describes the trend in the data](images/best_fit_line.png)

Picture a simpler example first: time spent studying versus exam score. Plenty of different lines could be drawn through that cloud of points, but most of them clearly miss the trend. The chart above shows two poor attempts in gray, and the actual best fit line, the one Linear Regression solves for mathematically rather than by eye, in solid color. The whole rest of this note is really about how that one best line gets found, precisely, for six features at once instead of just one.

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

Plugging in the very first row of real data, where a 19-year-old female, BMI 27.9, no children, a smoker, from the southwest region, was actually charged 16,884.92:

```
h(x1) = θ0 + θ1*19 + θ2*female + θ3*27.900 + θ4*0 + θ5*yes + θ6*southwest
y1 = 16884.92400
```

The model's whole job is to find values of `θ0` through `θ6` so that `h(x1)` comes out as close as possible to 16,884.92, and simultaneously as close as possible to every other `yi` in the dataset too, all at once. That balancing act across every single row is exactly what the cost function, covered shortly, measures.

![With two features, the best fit line becomes a plane](images/hyperplane.png)

This is what happens visually the moment you add a second feature: the single best fit line becomes a flat plane cutting through 3D space, minimizing the vertical distance to every point simultaneously across both features at once. With six features, as in our actual dataset, this same plane becomes a hyperplane, a flat surface in six-dimensional space that cannot be drawn, but works on exactly the same principle as the plane shown above.

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

![Residuals: the vertical distances between actual and predicted values](images/residuals.png)

The red lines in the chart above are exactly what the cost function is measuring: the vertical distance between each actual data point and where the line predicts it should be. These distances are called residuals. Notice they are not measured diagonally to the nearest point on the line, they are measured straight up or down, since we are trying to predict `y` from `x`, not find the closest point in any direction. The cost function squares every one of these red distances and averages them, exactly as explained above.

You will also often see this same quantity called the RSS, short for Residual Sum of Squares, usually referring to the sum of squared errors without dividing by `m`. `J(θ)` here is just the RSS averaged across all rows, and some textbooks or tutorials use `θ` for the parameter vector while others use `β` (beta) instead. Both refer to the exact same thing: the vector of parameters the model is trying to solve for.

To make the matrix version of this cost function work out cleanly, one small trick gets added: an extra column of all 1s gets bolted onto the front of `X`, corresponding to `θ0`, the intercept term. This turns `X` into a matrix of shape `(m, n+1)` instead of `(m, n)`. With that adjustment, the cost function can be written entirely in matrix form:

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

(The middle terms combine into one because `yᵀXθ` and `θᵀXᵀy` are transposes of each other, and since both are just single numbers, a 1x1 matrix, a number always equals its own transpose, so the two terms are identical and can be combined.)

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

This is the Normal Equation. Plug in your actual `X` matrix and `y` vector from the dataset, and this formula spits out the exact optimal values for every single `θ`, all at once, with no guessing, no trial and error, and no iteration required. It is a direct, analytical solution, which is exactly why it is considered one of the most elegant results in introductory machine learning.

## Fitting a Quick Visual: Charges vs BMI

Before judging any model numerically, it always helps to see the relationship visually first. Plotting just BMI against charges, ignoring the other five features for a moment, and fitting a regression line through it:

```python
sns.lmplot(x='bmi', y='charges', data=df, aspect=2, height=6)
plt.xlabel('Body Mass Index (kg/m²): Independent variable')
plt.ylabel('Insurance Charges: Dependent variable')
plt.title('Charges vs BMI')
```

This produces a scatter plot of every customer's BMI against their charges, with a single straight line drawn through the cloud of points, exactly the kind of best fit line the Normal Equation calculates directly, just restricted here to one feature so it can actually be drawn on a 2D chart.

## How Do You Know If the Model Is Any Good? Evaluation Metrics

Once the model produces predictions, you need a way to judge how close those predictions are to reality. Four metrics come up constantly for this, and each one tells a slightly different part of the story.

### Mean Absolute Error (MAE)

```
MAE = (1/N) * Σ |yi - ŷi|
```

MAE is the average of the absolute difference between the actual value `yi` and the predicted value `ŷi`. Taking the absolute value matters here: without it, a prediction that's 500 too high and another that's 500 too low would cancel each other out to zero on averaging, hiding the fact that the model was wrong both times. An MAE of 0 means perfect predictions every time; the larger the MAE, the worse the model.

MAE is measured in the same units as the original data (dollars, in our case), which makes it easy to interpret directly. It is also robust to outliers, meaning a single wildly-off prediction does not distort it too much, since errors are not squared, just averaged directly. The tradeoff is that MAE is not differentiable at zero, which becomes relevant if you were training a model using calculus-based optimizers like gradient descent instead of the direct Normal Equation used here.

### Mean Squared Error (MSE)

```
MSE = (1/N) * Σ (yi - ŷi)²
```

This is exactly the same calculation used inside the cost function `J(θ)` above. MSE squares every error before averaging, which has two effects: it removes the issue of positive and negative errors canceling out, and it punishes large errors disproportionately harder than small ones, since squaring a big number grows much faster than squaring a small one. This means MSE is quite sensitive to outliers, a single very bad prediction can inflate it substantially, which is the opposite tradeoff from MAE. Lower MSE always means a better fitting model.

### Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √[(1/N) * Σ (yi - ŷi)²]
```

RMSE simply takes the square root of MSE. Why bother? Because MSE's units are squared, dollars squared, in our case, which is not something you can intuitively interpret. Taking the square root brings the error back into the original, interpretable unit (plain dollars), while still keeping MSE's characteristic of penalizing large errors heavily. Lower RMSE means better predictions, and a noticeably higher RMSE than MAE on the same model is usually a sign that some predictions are wildly off, since RMSE reacts so strongly to large individual errors.

![How one outlier affects MAE, MSE, and RMSE differently](images/metrics_outlier_sensitivity.png)

The chart above makes the difference concrete. Starting from the same small set of errors, adding just one large outlier barely moves MAE, since it only averages absolute values, but it visibly inflates both MSE and RMSE, since squaring a large error makes it dominate the average. This is exactly why comparing MAE against RMSE on the same model is a quick, practical way to check for outliers: if RMSE is much larger than MAE, a handful of unusually bad predictions are likely dragging it up.

### R² Score (Coefficient of Determination)

```
R² = 1 - [Σ(yi - ŷi)² / Σ(yi - ȳ)²]
```

Here `ȳ` (read "y-bar") is simply the mean of all the actual `y` values. R² answers a different question than the previous three metrics: instead of measuring raw error size, it measures how much better your model is compared to just guessing the average every time.

The denominator, `Σ(yi - ȳ)²`, represents the total variation in the data if you made no effort at all and just predicted the average charge for everyone. The numerator represents the actual leftover error your model still has. R² typically ranges from 0 to 1: an R² of 1 means your model explains all the variation in the data perfectly. An R² of 0 means your model is doing no better than just guessing the average every single time. A negative R² is actually possible too, and it signals something has gone genuinely wrong, since it means your model is performing worse than that trivial "just guess the average" baseline.

## Bringing It All Together

Step back and look at the full journey this note just walked through. You started with a completely relatable, concrete problem: pricing insurance based on six pieces of information about a person. You expressed that problem as a hypothesis function, first for one feature, then generalized to all six. You packed that hypothesis into clean matrix notation so it could describe all 1,338 rows simultaneously with a single equation, `h(x) = Xθ`. You defined a cost function, `J(θ)`, to measure exactly how wrong any particular choice of parameters is. You then used basic calculus, setting the derivative of that cost function to zero, to derive the Normal Equation, `θ = (XᵀX)⁻¹Xᵀy`, a direct formula that hands you the best possible parameters in one step. And finally, you learned four different lenses, MAE, MSE, RMSE, and R², for judging whether the resulting model's predictions are actually trustworthy.

That is the complete arc of classical Linear Regression, from a plain English problem statement all the way through to a mathematically justified solution and a way to grade its own performance. Every other regression technique you learn from here, regularized regression, polynomial regression, even parts of neural networks, builds directly on top of this exact same foundation.

## Building Linear Regression From Scratch in Python, No Library Doing the Math for You

Everything above has been leading to one thing: actually writing code that computes `θ = (XᵀX)⁻¹Xᵀy` directly, with no gradient descent, no iteration, no guessing. This is called the closed form solution, since it is a direct, algebraic formula rather than something you approach step by step. To prove it genuinely works, and not just in theory, the code below is tested against `LinearRegression` from scikit-learn afterward, and both should produce the exact same result.

### Writing the Class

```python
import numpy as np

class LinearRegressionClosed:
    def __init__(self):
        # these get learned during fit(), so they start out empty
        self.coef_ = None       # this will hold theta1 ... thetan (the slopes)
        self.intercept_ = 0.0    # this will hold theta0 (the bias/intercept)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # prepend a column of 1s to X, exactly the x0 = 1 trick from the
        # hypothesis function section earlier in this note. This lets the
        # intercept theta0 get solved for using the same single formula,
        # instead of handling it as a separate special case.
        ones_column = np.ones((X.shape[0], 1))
        Xb = np.concatenate([ones_column, X], axis=1)

        # this one line IS the Normal Equation derived earlier:
        # theta = (X^T X)^-1 X^T y
        beta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T).dot(y)

        self.intercept_ = beta[0]    # the first value is theta0
        self.coef_ = beta[1:]         # everything after that is theta1...thetan

    def predict(self, X):
        X = np.array(X)
        # h(x) = X*theta, written here as a dot product plus the intercept
        return X.dot(self.coef_) + self.intercept_
```

Notice how directly this maps back to the math from earlier in this note. The `fit` method is nothing more than the Normal Equation, `θ = (XᵀX)⁻¹Xᵀy`, translated line by line into NumPy: `np.linalg.inv(...)` for the matrix inverse, `.dot(...)` for matrix multiplication, and `.T` for the transpose. The `predict` method is just `h(x) = Xθ`, the hypothesis function, applied using the coefficients that `fit` just solved for.

### Testing It Against a Real Dataset

To confirm this actually works, and not just for our own custom insurance data, here it is applied to the California Housing dataset, and compared directly against scikit-learn's own, professionally implemented `LinearRegression`.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# load a real dataset: predicting median house value from housing features
X, y = fetch_california_housing(return_X_y=True)

# fit and predict using OUR from-scratch implementation
reg = LinearRegressionClosed()
reg.fit(X, y)
y_pred = reg.predict(X)
print("Our from-scratch R2 score:", r2_score(y, y_pred))

# fit and predict using scikit-learn's implementation, for comparison
reg_sk = LinearRegression()
reg_sk.fit(X, y)
y_pred_sk = reg_sk.predict(X)
print("scikit-learn R2 score:", r2_score(y, y_pred_sk))
```

```
Our from-scratch R2 score: 0.6062326851998051
scikit-learn R2 score: 0.6062326851998051
```

Both scores come out identical, down to the decimal. This is a genuinely satisfying confirmation: the Normal Equation derived earlier in this note, by hand, using nothing but partial derivatives and a handful of algebra steps, is exactly the same calculation running quietly inside a production-grade, widely used library like scikit-learn. There is no hidden trick scikit-learn is using instead. It is this same formula, just written by people who have also handled the extra engineering details, like numerical stability for very large datasets, that a from-scratch version like this one does not worry about.

### Why This Version Doesn't Use Gradient Descent

It is worth being explicit about what makes this approach different from the more commonly taught gradient descent version of Linear Regression. Gradient descent starts with a random guess for `θ` and nudges it a little closer to the optimal answer on every iteration, repeating that process many times until the parameters settle down. The closed form solution used here skips all of that entirely: since the Normal Equation is a direct, exact formula, one single calculation hands you the true optimal `θ`, with no iterations and no risk of stopping early before reaching the best answer.

![Gradient descent vs the closed form solution](images/gd_vs_closedform.png)

The chart above shows this difference directly on a simple cost curve. On the left, gradient descent starts at a random point and takes repeated small steps downhill, getting closer to the minimum with each one, but never landing on it exactly in a finite number of steps. On the right, the closed form solution ignores the starting point entirely and jumps straight to the minimum in a single calculation, since the Normal Equation solves for it directly rather than approaching it gradually.

The tradeoff, not covered in depth in this note but worth knowing, is that computing a matrix inverse, `(XᵀX)⁻¹`, becomes computationally expensive once you have a very large number of features, which is exactly the kind of situation where gradient descent, despite needing many iterations, ends up being the more practical choice instead.
