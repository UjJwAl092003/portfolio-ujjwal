# Linear Algebra for Machine Learning, Part 1: Vectors, Norms and the Foundations of Matrix Calculus

This is the first part in a linear algebra series. It does not cover all of linear algebra, and it is not meant to. Topics like eigenvalues, eigenvectors, and matrix decomposition are big enough to deserve their own separate note later, and they will come in a future part of this series.

What this part does cover is the foundation everything else sits on top of. By the end of this note, you will understand how data becomes numbers a machine can work with, how to measure the size and distance of that data, and how derivatives work when your input is a vector or a matrix instead of a single number.

## What You Will Be Able to Understand After This

Being honest about scope matters in math, so here is exactly what this note prepares you for.

After finishing this part, you will be able to properly understand the math behind linear regression, both the gradient descent version and the normal equation version, since both rely directly on the quadratic form derivative covered here. You will understand why gradient descent works the way it does, since gradients are explained from scratch. You will understand distance based methods like K Nearest Neighbours, since those rely entirely on norms. You will understand the intuition behind L1 and L2 regularization, since those penalty terms are literally the norms covered in this note. You will also be comfortable reading most beginner to intermediate machine learning papers and courses when they mention gradients, vectors, or dot products.

What you will not yet fully understand is anything that depends on eigenvalues or eigenvectors, such as Principal Component Analysis, Singular Value Decomposition, or spectral clustering. Those need a dedicated part of their own, since they introduce a genuinely new way of thinking about matrices. Consider this note the ground floor. The next parts will build the upper floors on top of it.

## Why Machine Learning Needs Linear Algebra at All

A machine learning algorithm can only work with numbers. It has no concept of what a picture, a sound, or a sentence actually is. So before any learning can happen, everything has to be converted into numbers first.

This is the difference between qualitative data and quantitative data. Qualitative data is something like an image, a voice recording, a color, or a smell. None of these can be fed into an algorithm directly. Quantitative data is something like a height of 170, a temperature of 37, or a pixel brightness of 255. Machine learning algorithms only ever work with quantitative data, which means the very first job in any ML pipeline is turning everything qualitative into something quantitative.

### A Concrete Example: How a Computer Sees a Handwritten Digit

Take a handwritten digit like the number 8. A human looks at it and instantly recognizes the shape. A computer does not see a shape at all. It sees a grid of numbers, where each number represents how bright or dark that tiny part of the image is.

So the journey looks like this.

```
Image of a digit
        |
Matrix of pixel brightness values
        |
Vector of numbers
        |
Machine Learning Algorithm
```

This exact idea is the foundation of the MNIST dataset, one of the most famous datasets in machine learning, made up of thousands of handwritten digits from 0 to 9. The task is simple to describe: given an image as input, predict which digit it is. This is called a classification problem, and it only becomes solvable once the image is turned into numbers.

### How an Image Actually Becomes a Matrix

Say an image is 60 pixels wide and 60 pixels tall. Every single pixel holds one number. For a grayscale image, that number usually ranges from 0, which means pure black, to 255, which means pure white. Everything in between is a shade of grey.

So a grayscale image is really nothing more than a 60 by 60 matrix of numbers.

### Flattening a Matrix Into a Vector

Many machine learning algorithms expect their input as a single list of numbers, called a vector, rather than a grid. So the matrix gets unrolled, or flattened, into one long vector.

For example, a small matrix like this

```
1  2
3  4
```

can be unrolled into a vector like this

```
1
3
2
4
```

The exact order depends on the implementation, but the idea stays the same. A 60 by 60 image, once flattened, becomes a vector with 3600 numbers in it, since 60 multiplied by 60 is 3600.

### Color Images and Video Need One More Idea

A color image is not just one matrix, it is three matrices stacked together, one each for red, green and blue. So a color image of size 600 by 538 pixels is represented as 600 by 538 by 3. This is no longer a simple 2D matrix, it is called a tensor, since it has more than two dimensions.

A video takes this one step further. A video is simply a sequence of images, so if a video has 100 frames, its shape becomes 100 by height by width by 3, giving four dimensions total instead of three. This is why the word tensor shows up constantly in deep learning, and it is also why Google's deep learning library is literally named TensorFlow.

## Scalars, Vectors, Matrices and Tensors

These four words describe data of increasing complexity, and getting comfortable with the differences between them will make the rest of this note, and most ML material you read afterward, much easier to follow.

**Scalar.** A single number, nothing more. Examples include a learning rate, a bias term, or a temperature reading. In terms of order, a scalar is called a 0th order tensor.

**Vector.** A collection of numbers arranged in a single column or row. For example, a vector representing three feature values might look like this.

```
x = [2, 5, 7]
```

Vectors are usually written in bold lowercase letters, like **x**. A vector is a 1st order tensor. The number of entries inside a vector is called its dimension, which is a completely different idea from its order. A vector with four numbers in it has dimension 4, but its order is still just 1. This distinction trips up a lot of beginners, so it is worth remembering deliberately: order tells you how many indices you need to locate a value, dimension tells you how many values exist along one of those indices.

**Matrix.** A two dimensional grid of numbers, written in bold uppercase, like **A**. For example

```
A = [1  2]
    [3  4]
```

A matrix is a 2nd order tensor.

**Tensor.** Anything with an order greater than two. A color image, which is height by width by 3, is a tensor. A video, which adds a time dimension on top of that, is also a tensor.

### Why Machine Learning Uses Such High Dimensional Vectors

In physics, you are used to three dimensional space, described by x, y and z. Machine learning routinely works with vectors that have thousands, or even millions, of dimensions. A flattened 60 by 60 image alone already gives you a 3600 dimensional vector. This is completely normal in ML, even though it is impossible to visualize directly.

### Learning as a Vector Transformation

Here is one of the most useful mental models in all of machine learning. Suppose you have an input vector v1, and you want to produce an output vector v2. This transformation can be written as

```
v2 = W * v1
```

where W is a matrix. Learning, in a very real sense, means finding the correct matrix W that turns your inputs into the outputs you actually want. Almost every machine learning model, from a simple linear regression to a deep neural network, is ultimately searching for the right transformation matrices.

### Why Turning Everything Into Vectors Is So Powerful

Once you represent images, sounds, words, or anything else as vectors, something useful happens. Similar things end up geometrically close to each other in that vector space, while different things end up far apart. Imagine plotting vectors representing cats, dogs and cars on a graph. Cats would cluster together in one region, dogs in another, and cars somewhere else entirely. Classification, at its core, becomes the task of drawing boundaries that separate these clusters.

## Norms: Measuring the Size of a Vector

### What a Norm Actually Is

A scalar like 5 or minus 3 has an obvious size. A vector does not, at least not automatically, since it has multiple components. A norm is the tool that takes a vector and converts it into a single positive number representing its size, magnitude, or length. It is written as ||x||.

### Why Norms Matter in Machine Learning

There are two main reasons you constantly run into norms in ML.

The first is measuring the size of a single vector. For example, the vector (3, 4) has a length of 5, which you can verify using the same idea as the Pythagorean theorem.

The second, and arguably more important reason, is measuring similarity between two vectors. Suppose you have two images, Image A and Image B, both represented as vectors. Subtracting one from the other gives you a difference vector, A minus B. Taking the norm of that difference tells you how similar or different the two images are. A small norm means the images are similar. A large norm means they are quite different. This exact idea is what powers algorithms like K Nearest Neighbours, where the entire prediction is based on finding the closest points using a norm.

### The Rules Every Valid Norm Must Follow

Not just any formula qualifies as a norm. A valid norm must satisfy three properties.

First, the norm of a vector is zero only when the vector itself is entirely zero. A vector with any nonzero component cannot have a norm of zero.

Second, the triangle inequality must hold, meaning ||x + y|| is always less than or equal to ||x|| + ||y||. This is really just saying that a straight line between two points is never longer than any indirect path between them.

Third, scaling a vector scales its norm by the same factor, so ||alpha times x|| always equals |alpha| times ||x||. Doubling a vector doubles its length, which matches common sense.

### The Euclidean Norm, Also Called the L2 Norm

This is the most commonly used norm, and it matches your everyday intuition of length or distance. The formula is

```
||x||_2 = square root of (x1^2 + x2^2 + ... + xn^2)
```

For example, take the vector (-5, 3, 2). Squaring each value and adding them gives 25 + 9 + 4 = 38. Taking the square root of 38 gives approximately 6.16, which is the Euclidean norm of that vector.

### The L1 Norm

The L1 norm simply adds up the absolute values of every component, without squaring anything.

```
||x||_1 = |x1| + |x2| + ... + |xn|
```

For the same vector (-5, 3, 2), the L1 norm is 5 + 3 + 2 = 10.

### The General p-Norm

Both L1 and L2 are actually special cases of a more general formula, called the p-norm.

```
||x||_p = (sum of |xi|^p)^(1/p)
```

This formula is only valid when p is greater than or equal to 1. Setting p equal to 1 gives you the L1 norm, and setting p equal to 2 gives you the Euclidean norm.

### The Infinity Norm

Also called the maximum norm, this one simply picks out the single largest absolute value among all the components, and ignores everything else.

```
||x||_infinity = max(|xi|)
```

For (-5, 3, 2), the largest absolute value is 5, so the infinity norm equals 5.

### Norms for Matrices: The Frobenius Norm

Norms are not limited to vectors, matrices can have a norm too. The most common one is the Frobenius norm, which is calculated by squaring every entry in the matrix, adding them all up, and taking the square root, essentially treating the whole matrix like one long flattened vector.

```
||A||_F = square root of (sum of every entry squared)
```

For example, take the matrix

```
A = [1  2]
    [2  0]
```

Squaring and adding every entry gives 1 + 4 + 4 + 0 = 9, and the square root of 9 is 3, so the Frobenius norm of A is 3.

### Where You Will Actually See Norms Used

Norms show up constantly once you start working with real machine learning code, in gradient descent, in optimization more generally, in measuring prediction error, in comparing image similarity, in loss functions, and especially in regularization techniques like L1 and L2 regularization, which literally add a norm of the weight vector to the loss function to discourage overly large weights.

## Linear Combinations, Span, and Linear Independence

### Linear Combination

If you take a set of vectors and multiply each one by some scalar, then add the results together, you get a linear combination.

```
alpha1 * v1 + alpha2 * v2 + ... + alphan * vn
```

For example, take

```
v1 = [1, 2, 3]
v2 = [2, 0, 3]
```

Computing v1 + 2 times v2 gives

```
[1 + 4, 2 + 0, 3 + 6] = [5, 2, 9]
```

This idea directly connects to matrix multiplication. If a matrix A is made of columns v1 and v2 side by side, then multiplying A by the vector (1, 2) gives exactly v1 + 2 times v2. In other words, matrix multiplication is really just a compact way of writing a linear combination of columns. This interpretation turns out to be extremely useful throughout machine learning.

### Span

The span of a set of vectors is simply every possible linear combination you could ever build from them. For example, the vectors (1, 0) and (0, 1) can be combined, using different scalar multiples, to reach any point (x, y) in two dimensional space. So their span is the entire 2D plane.

### Column Space

The span of all the columns of a matrix is called its column space. This idea matters because if you are trying to solve an equation like Ax = b, that equation only has a solution if b happens to lie inside the column space of A. If b falls outside that space, no choice of x can ever produce it.

### Linear Independence

A set of vectors is linearly independent if none of them can be built as a combination of the others. For example, (1, 0) and (0, 1) are linearly independent, since there is no way to construct either one using only the other.

On the other hand, take (1, 0), (0, 1) and (3, 4). Here, (3, 4) can be written as 3 times (1, 0) plus 4 times (0, 1), so this third vector is redundant. It adds no new direction that the first two did not already cover. This makes the set linearly dependent.

The formal test for independence is this. A set of vectors is independent only if the equation

```
alpha1 * v1 + alpha2 * v2 + ... + alphak * vk = 0
```

has no solution other than every alpha being zero. If some other combination of non-zero alphas also produces zero, the vectors are dependent.

## Matrix Calculus: Taking Derivatives Beyond Single Numbers

Ordinary calculus teaches you how to differentiate a function of a single number with respect to another single number. Matrix calculus extends this same idea to vectors and matrices, and it becomes essential the moment you start training any real machine learning model.

### Why This Is Needed in the First Place

Every machine learning model is essentially learning a mapping from an input vector to an output vector. During training, the model's predictions are usually wrong at first, so its parameters, meaning its weights, need to be adjusted so the output improves. The natural question is: if I nudge one parameter slightly, how much does the output change? Answering that question is exactly what derivatives are for, except now the derivative needs to work with vectors and matrices instead of single numbers.

### Case 1: Differentiating a Vector With Respect to a Scalar

Suppose you have a vector whose entries are each a function of a single scalar x.

```
a = [x^2, x^3, x^5]
```

To differentiate this vector with respect to x, you simply differentiate each entry individually.

```
da/dx = [2x, 3x^2, 5x^4]
```

The result of differentiating a vector with respect to a scalar is simply another vector of the same size.

### Case 2: Differentiating a Scalar With Respect to a Vector, the Gradient

This is one of the single most important ideas in all of machine learning, since gradient descent is built entirely on top of it.

Suppose you have a function of three variables.

```
f(x, y, z) = x * y * z^2
```

Group the inputs into a vector x = (x, y, z). The derivative of this scalar output with respect to the input vector is called the gradient, written as the symbol nabla f, and it is simply a vector containing the partial derivative with respect to each variable.

```
gradient of f = [df/dx, df/dy, df/dz]
```

Working through the example, the partial derivative with respect to x is y times z squared, the partial derivative with respect to y is x times z squared, and the partial derivative with respect to z is 2 times x times y times z. So the full gradient is

```
gradient of f = [y*z^2, x*z^2, 2*x*y*z]
```

The gradient tells you two crucial things. It tells you the direction in which the function increases the fastest, and it tells you how sensitive the output is to each individual input variable. This is precisely why gradient descent works the way it does: at every step, the algorithm looks at the gradient of the loss function and moves the parameters in the exact opposite direction, since that is the direction of fastest decrease.

### A Note on Layout: Row Vector or Column Vector

Before moving to the next case, it helps to clear up something that confuses a lot of people when they start reading different books or courses on this topic. The three cases covered here all follow what is called the numerator layout convention, which simply means the shape of the result is decided by the shape of the thing on top of the fraction, before you even look at what is on the bottom.

Following that convention strictly, differentiating a vector with respect to a scalar gives you a column vector, since the vector on top has n entries stacked vertically. This matches the position vector example above exactly.

Differentiating a scalar with respect to a vector, strictly following numerator layout, actually gives you a row vector instead of a column vector, since the scalar on top has no shape of its own, so the shape comes entirely from how you arrange the entries of the vector on the bottom, listed side by side.

```
dy/dx = [dy/dx1   dy/dx2]
```

In practice, however, machine learning and optimization material almost always writes this same derivative as a column vector instead, and calls it the gradient. This is simply the transpose of the strict row vector version above, and both contain the exact same information, just arranged differently.

```
gradient = [dy/dx1]
           [dy/dx2]
```

This is exactly why the gradient in this note is written as a column vector throughout. If you come across another source that writes it as a row, do not worry, it is the same derivative, just laid out the other way. Once you know this convention exists, switching between different textbooks and courses becomes far less confusing.

Differentiating a vector with respect to another vector, where the top vector has m entries and the bottom vector has n entries, produces a full matrix of shape m by n, which is exactly the Jacobian matrix covered next.

### Case 3: Differentiating a Vector With Respect to Another Vector, the Jacobian Matrix

Suppose vector **a** depends on vector **b**. Differentiating every entry of **a** with respect to every entry of **b** produces a full matrix, not just a vector, called the Jacobian matrix. Each entry in the Jacobian tells you how one specific output component changes with respect to one specific input component.

A simple physical example helps here. Imagine the velocity of air at a point in space, which depends on the x, y and z position. The Jacobian matrix in this case would tell you exactly how each component of velocity changes as you move slightly in any of the three spatial directions.

The key pattern to remember across all three cases so far: differentiating a vector with respect to a scalar gives you a vector, differentiating a scalar with respect to a vector gives you a vector called the gradient, and differentiating a vector with respect to another vector gives you a full matrix called the Jacobian.

### The Derivative of a Dot Product

This particular result comes up so often in machine learning that it is worth memorizing directly. If you have two vectors **x** and **a**, their dot product is a single scalar, computed as

```
x_transpose * a = a1*x1 + a2*x2 + a3*x3
```

Differentiating this scalar with respect to the vector **x** simply gives you back the vector **a** itself.

```
d(x_transpose * a) / dx = a
```

This makes intuitive sense once you look at the individual partial derivatives. The derivative with respect to x1 is just a1, the derivative with respect to x2 is just a2, and so on, so stacking these back into a vector gives you exactly **a** again.

### The Product Rule for Matrices

If two matrices A and B both depend on some variable x, differentiating their product follows a rule that looks similar to the ordinary product rule from single variable calculus, with one very important twist.

```
d(AB)/dx = (dA/dx) * B + A * (dB/dx)
```

The critical rule here is that you must never swap the order of multiplication. Matrix multiplication is not commutative, meaning AB does not generally equal BA, so writing the terms in the wrong order gives you a completely wrong result. Always keep dA/dx on the left when it multiplies B, and keep A on the left when it multiplies dB/dx.

### The Quadratic Form, and Its Derivative

One of the most frequently seen expressions in machine learning is the quadratic form, written as

```
Q = x_transpose * A * x
```

Here x is an n by 1 vector and A is an n by n matrix. Multiplying these together in this exact order, transpose of x times A times x, always produces a single scalar, since a 1 by n matrix times an n by n matrix times an n by 1 matrix results in a 1 by 1 output.

The derivative of this quadratic form with respect to x is one of the most important formulas you will ever use in machine learning.

```
d(x_transpose * A * x) / dx = (A + A_transpose) * x
```

And in the very common special case where A is symmetric, meaning A equals its own transpose, this formula simplifies neatly to

```
d(x_transpose * A * x) / dx = 2 * A * x
```

This single formula is what makes it possible to solve linear regression directly using the normal equation, and it shows up again inside the derivation of least squares, and in various neural network calculations.

### Quick Verification With a 2 by 2 Example

To see why this formula is actually true, and not just something to memorize blindly, it helps to expand a small example by hand. Take a general 2 by 2 matrix A with entries A11, A12, A21 and A22, and a vector x with entries x1 and x2. Expanding Q = x*transpose * A \_ x gives

```
Q = A11*x1^2 + A12*x1*x2 + A21*x2*x1 + A22*x2^2
```

Differentiating this expression with respect to x1 gives 2*A11*x1 + (A12 + A21)*x2, and differentiating with respect to x2 gives (A12 + A21)*x1 + 2*A22*x2. Stacking these two partial derivatives back into a vector, and comparing it against (A + A_transpose) times x, confirms that the general formula matches exactly. This is a good exercise to work through yourself on paper at least once, since it turns the formula from something memorized into something understood.

### Why All of This Matters for Training a Model

Every machine learning algorithm is, at its core, trying to minimize some loss function, whether that is Mean Squared Error, Logistic Loss, or Cross Entropy. Minimizing that loss requires computing its derivative with respect to every single weight in the model. The standard update rule used during training looks like this.

```
w = w - learning_rate * gradient_of_loss(w)
```

Here w represents the entire vector of parameters, and the gradient tells the algorithm exactly which direction to nudge those parameters to reduce the loss. Without matrix calculus, deriving update rules like this one for anything beyond the simplest models would be extremely difficult, since real models have thousands or millions of parameters bundled together as vectors and matrices, not single numbers.

## Summary Table

| Concept                   | What It Means                                                              |
| ------------------------- | -------------------------------------------------------------------------- |
| Scalar                    | A single number, 0th order tensor                                          |
| Vector                    | A list of numbers, 1st order tensor                                        |
| Matrix                    | A 2D grid of numbers, 2nd order tensor                                     |
| Tensor                    | Any array with order greater than 2                                        |
| Flattening                | Converting a matrix or image into a single vector                          |
| Norm                      | A single number representing a vector's size or distance                   |
| L1 Norm                   | Sum of absolute values of each component                                   |
| L2 Norm                   | Ordinary Euclidean length                                                  |
| Infinity Norm             | The single largest absolute component                                      |
| Frobenius Norm            | The matrix equivalent of the L2 norm                                       |
| Linear Combination        | A weighted sum of vectors                                                  |
| Span                      | Every possible linear combination of a set of vectors                      |
| Column Space              | The span of a matrix's columns                                             |
| Linear Independence       | No vector in the set can be built from the others                          |
| Gradient                  | Derivative of a scalar with respect to a vector                            |
| Jacobian                  | Derivative of a vector with respect to another vector                      |
| Quadratic Form Derivative | d(x transpose A x)/dx equals (A + A transpose) x, or 2Ax if A is symmetric |

## What Comes Next

This note covers enough linear algebra to properly understand linear regression, gradient descent as a general optimization technique, distance based algorithms like K Nearest Neighbours, and the reasoning behind L1 and L2 regularization. It intentionally leaves out eigenvalues, eigenvectors, and matrix decomposition, since those ideas deserve a focused note of their own and are needed for topics like Principal Component Analysis and dimensionality reduction.

The plan going forward is to cover probability next, since it forms the other major mathematical pillar underneath machine learning, followed by core ML algorithms themselves, and then a dedicated part covering the more advanced side of linear algebra, including eigenvalues and decomposition. Each part is meant to stand on its own while building toward a complete picture.
