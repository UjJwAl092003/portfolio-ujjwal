# NumPy for Machine Learning: A Complete Beginner's Guide

Before you can do anything in machine learning, whether that is loading a dataset, running matrix calculus, or feeding numbers into a model, you need a way to store and manipulate large collections of numbers efficiently. That tool is NumPy, and this guide walks through it from the very beginning, assuming you have never used it before.

## What is NumPy and Why It Matters in Machine Learning

NumPy, short for Numerical Python, is a library built for working with numbers, vectors, and matrices efficiently in Python. It matters so much in machine learning because almost every other data and ML library in Python, including Pandas, Scikit-learn, TensorFlow, and PyTorch, is either built directly on top of NumPy or designed to work smoothly with it. Learning NumPy properly means you are learning the common language that the entire Python data science ecosystem speaks.

There is also a very practical reason NumPy is preferred over plain Python lists. NumPy is written with bindings to C, a much faster low level language, which makes operations on NumPy arrays significantly faster than the equivalent operations on regular Python lists, especially once your data gets large. When you are working with an image made of thousands of pixels, or a dataset with millions of rows, this speed difference becomes the difference between code that runs instantly and code that takes forever.

## Installing and Importing NumPy

If you do not already have NumPy installed, you can install it using pip.

```
pip install numpy
```

Once installed, the standard convention across almost all Python code you will ever see is to import it with the short alias `np`.

```python
import numpy as np
```

From here on, every example uses this `np` alias.

## NumPy Arrays: The Core Building Block

Everything in NumPy revolves around one central object called the array. Arrays come in two flavors that you will use constantly.

A vector is a strictly one dimensional array, essentially a single row of numbers.

A matrix is a two dimensional array, essentially a grid of rows and columns. It is worth noting that a matrix can still technically have only one row or one column and still be considered two dimensional, which is a subtlety that trips up a lot of beginners later when reshaping arrays.

### Creating an Array From a Python List

The simplest way to create a NumPy array is to convert an existing Python list directly.

```python
my_list = [1, 2, 3]
np.array(my_list)
```

```
array([1, 2, 3])
```

The same idea extends to a list of lists, which becomes a two dimensional array, effectively a matrix.

```python
my_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np.array(my_matrix)
```

```
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

### The ndarray Type

Every array you create in NumPy, whether it started as a list, a range, or random numbers, is actually an object of the same underlying type, called `ndarray`, short for n-dimensional array. You can check this yourself.

```python
a = np.array([1, 2, 3, 4, 5])  # create a simple 1D array
type(a)                        # check what type of object this is
```

```
numpy.ndarray
```

An `ndarray` is a fixed size array stored in memory where every single value must be of the same data type, for example all integers or all floating point numbers. This fixed size, single data type design is exactly what makes NumPy arrays so much faster than regular Python lists, since Python does not need to check the type of every individual element one by one during an operation.

### Combining Arrays: vstack and hstack

Sometimes you already have two separate arrays and you want to combine them into one. NumPy gives you two direct ways to do this depending on whether you want to stack them vertically or horizontally.

`vstack` stacks arrays on top of each other, adding new rows.

```python
a = np.array([1, 2, 3, 4, 5])    # first array
b = np.array([6, 7, 8, 9, 10])   # second array

np.vstack((a, b))                # stack a and b as two rows
```

```
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10]])
```

`hstack` does the opposite, it stacks arrays side by side, adding new columns instead of new rows.

```python
np.hstack((a, b))                # stack a and b side by side in one row
```

```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

You will use `vstack` and `hstack` often when you are assembling a dataset from separate pieces, for example combining a new batch of rows onto an existing training set.

## Built-in Ways to Generate Arrays

Converting a list works fine for small examples, but in real projects you will constantly need to generate arrays of specific sizes and patterns directly, without typing out every number by hand. NumPy provides several functions for exactly this.

### arange

`arange` returns evenly spaced values within a given range, working very similarly to Python's built-in `range`, except it returns a NumPy array instead of a plain list.

```python
np.arange(0, 10)
```

```
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

You can also specify a step size as a third argument.

```python
np.arange(0, 11, 2)
```

```
array([ 0,  2,  4,  6,  8, 10])
```

### zeros and ones

These two functions generate arrays filled entirely with zeros or entirely with ones, which is extremely common when initializing weights, masks, or placeholder arrays in machine learning code.

```python
np.zeros(3)
```

```
array([0., 0., 0.])
```

Passing a tuple instead of a single number creates a multi dimensional array of that shape.

```python
np.zeros((5, 5))
```

```
array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]])
```

The `ones` function works exactly the same way, just filling the array with 1 instead of 0.

```python
np.ones((3, 3))
```

```
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
```

### linspace

`linspace` also returns evenly spaced numbers, but unlike `arange`, where you specify a step size, with `linspace` you specify exactly how many numbers you want, and NumPy figures out the correct spacing between them to fit that many values evenly across the range.

```python
np.linspace(0, 10, 3)
```

```
array([ 0.,  5., 10.])
```

Asking for more points simply packs them more densely across the same range.

```python
np.linspace(0, 10, 50)
```

This produces 50 evenly spaced values starting at 0 and ending at 10.

### eye

`eye` creates an identity matrix, a square matrix with 1s along the diagonal and 0s everywhere else. Identity matrices show up constantly in linear algebra and machine learning, often as a starting point for certain matrix operations or as part of regularization terms.

```python
np.eye(4)
```

```
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
```

## Generating Random Arrays

Machine learning code constantly needs random numbers, whether to initialize model weights, shuffle data, or generate synthetic examples for testing. NumPy provides several different ways to generate randomness, and it matters which one you pick, since each follows a different underlying distribution.

### rand

`rand` generates numbers from a uniform distribution, meaning every value between 0 and 1 is equally likely to appear.

```python
np.random.rand(2)
```

```
array([0.23565463, 0.46336045])
```

Passing two numbers instead of one generates a two dimensional array of that shape.

```python
np.random.rand(5, 5)
```

This generates a 5 by 5 matrix where every value lies between 0 and 1, with no value more likely than any other.

### randn

`randn` generates numbers from the standard normal distribution instead, meaning values cluster around 0, with numbers farther from 0 becoming progressively less likely. This is different from `rand` in an important way, since `randn` can return negative numbers, while `rand` never can.

```python
np.random.randn(2)
```

```
array([-0.29129927, -1.27140728])
```

This distinction matters a lot in machine learning, since many weight initialization strategies for neural networks specifically rely on the normal distribution rather than the uniform one.

### randint

`randint` returns random whole numbers, or integers, between a lower bound that is included and an upper bound that is excluded.

```python
np.random.randint(1, 100)
```

```
62
```

You can also generate several random integers at once by passing a third argument specifying how many you want.

```python
np.random.randint(1, 100, 10)
```

```
array([80, 38, 51, 16, 90, 85, 53, 60, 18, 57])
```

## Useful Array Attributes and Methods

Once you have an array, whether created manually or generated randomly, there are several attributes and methods you will use constantly to understand and reshape it.

### reshape

`reshape` takes the same data inside an array and rearranges it into a new shape, as long as the total number of elements stays the same.

```python
arr = np.arange(25)
arr.reshape(5, 5)
```

```
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
```

This is one of the most frequently used operations in machine learning, since raw data often needs to be reshaped before it fits the exact input shape a model expects. Remember, though, that reshape only works if the new shape can hold exactly the same number of elements as before. You cannot reshape 25 elements into a 4 by 4 grid, since that would need 16 elements, not 25.

### Reshaping a 1D Array Into a Column, for Libraries Like Scikit-learn

A very specific reshaping pattern comes up constantly once you start using libraries like Scikit-learn. Many of them expect your target values, usually called `y`, to be shaped as a two dimensional array with exactly one column, not as a plain one dimensional array, even though both technically hold the same values.

```python
arr = np.array([11, 22, 33, 44, 55])       # a plain 1D array
print(arr.shape)                            # check its current shape

arr = arr.reshape((arr.shape[0], 1))         # reshape to (5 rows, 1 column)
print(arr.shape)                             # confirm the new shape
```

```
(5,)
(5, 1)
```

Using `arr.shape[0]` instead of typing the number 5 directly is a good habit, since it means the same line of code keeps working correctly even if the size of your array changes later.

### Reshaping a 2D Array Into a 3D Array

The same idea extends further when you need to add an extra dimension to a 2D array, which is common when preparing image data or sequence data for certain models.

```python
arr = [[11, 22], [33, 44], [55, 66]]         # a plain list of lists
arr = np.array(arr)                          # convert it into a NumPy array
print(arr.shape)                             # check the current 2D shape

arr = arr.reshape((arr.shape[0], arr.shape[1], 1))   # add a third dimension
print(arr.shape)                             # confirm the new shape
```

```
(3, 2)
(3, 2, 1)
```

Here the data itself has not changed at all, only how it is organized in terms of dimensions, which is exactly what reshaping is meant to do.

### max, min, argmax and argmin

These methods help you find the largest or smallest value inside an array, or, just as usefully, the index position where that value occurs.

```python
ranarr = np.random.randint(0, 50, 10)

ranarr.max()
ranarr.argmax()
ranarr.min()
ranarr.argmin()
```

`max` and `min` return the actual value, while `argmax` and `argmin` return the position, or index, at which that value is found. This distinction matters a lot in machine learning, for example when you have a list of predicted probabilities for different classes, and you specifically want to know which class index had the highest probability, not just what that highest probability value was.

### shape

`shape` is an attribute, not a method, meaning you access it without parentheses. It tells you the exact dimensions of an array.

```python
arr = np.arange(25)
arr.shape
```

```
(25,)
```

This single number in parentheses followed by a comma indicates a one dimensional array with 25 elements. Reshaping the same array into a row changes this shape entirely.

```python
arr.reshape(1, 25).shape
```

```
(1, 25)
```

And reshaping it into a column changes it again.

```python
arr.reshape(25, 1).shape
```

```
(25, 1)
```

Checking the shape of an array constantly, especially before feeding data into a machine learning model, is one of the most common debugging habits you will develop, since a huge number of errors in ML code come down to arrays having a shape the code did not expect.

### dtype

`dtype` tells you the data type of the values stored inside the array, such as integers or floating point numbers.

```python
arr.dtype
```

```
dtype('int64')
```

## Indexing and Selecting Elements

Being able to pull out specific elements, rows, or sections of an array is something you will do constantly, whether you are inspecting a dataset or slicing out a batch of training examples.

### Bracket Indexing and Slicing

For a one dimensional array, indexing looks almost identical to indexing a regular Python list.

```python
arr = np.arange(0, 11)

arr[8]
```

```
8
```

You can also grab a range of values using slicing, where the first number is included and the second number is excluded.

```python
arr[1:5]
```

```
array([1, 2, 3, 4])
```

### Negative Indexing in Slices

You can also count from the end of an array instead of the beginning, using negative numbers. An index of -1 refers to the last element, -2 refers to the second last element, and so on.

```python
arr = np.array([11, 12, 13, 14, 15])   # a simple 1D array
print(arr[-2:])                         # take the last two elements
```

```
[14 15]
```

Reading this slice out loud helps it stick. `-2:` means start at the second last element, and go all the way to the end, since no stopping index was given. This is extremely handy when you know how many elements you want from the end of an array, but do not want to bother calculating the exact positive index yourself.

### Broadcasting

One major way NumPy arrays differ from ordinary Python lists is their ability to broadcast a single value across a whole range of positions in one line.

```python
arr[0:5] = 100
```

```
array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10])
```

This single line replaced five separate values at once, something a plain Python list cannot do this directly.

### Broadcasting During Arithmetic

There is a second, even more common form of broadcasting that you will run into constantly, and it is worth understanding clearly. Normally, arithmetic between two arrays only works if both arrays are exactly the same shape. Broadcasting is the rule NumPy uses to bend this restriction, letting you combine arrays of different shapes in specific, predictable situations, the most common being combining an entire array with a single scalar number.

```python
A = np.array([
    [1, 2, 3],
    [1, 2, 3]
])                        # a 2 by 3 matrix

b = 2                     # a single scalar number

C = A + b                 # add the scalar to every element of A
print(C)
```

```
[[3 4 5]
 [3 4 5]]
```

Even though `A` is a matrix and `b` is just a single number, NumPy does not throw an error demanding matching shapes. Instead, it automatically stretches, or broadcasts, the scalar `2` across every single position in `A`, as if it had been an entire matrix of 2s all along. This is why the earlier example, `arr[0:5] = 100`, worked too, since assigning a single number into a slice relies on this exact same broadcasting rule.

### Views Versus Copies: An Important Gotcha

This next part catches almost every NumPy beginner at least once, so pay close attention. When you take a slice of a NumPy array, you do not get an independent copy of that data, you get a view into the original array. That means changing the slice also changes the original array.

```python
arr = np.arange(0, 11)
slice_of_arr = arr[0:6]

slice_of_arr[:] = 99
```

```
array([99, 99, 99, 99, 99, 99])
```

Now look at what happened to the original array.

```python
arr
```

```
array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])
```

The original array changed too, even though you only modified the slice directly. This behavior exists deliberately, since it avoids unnecessary duplication of data in memory, which matters a great deal when working with large datasets. If you specifically want an independent copy that will not affect the original array, you need to say so explicitly using `copy`.

```python
arr_copy = arr.copy()
```

Now changes to `arr_copy` will never affect `arr`, and vice versa.

### Indexing a Two Dimensional Array

For a matrix, you can index using either double brackets or a single set of brackets with a comma inside, though the comma style is generally clearer and preferred.

```python
arr_2d = np.array(([5, 10, 15], [20, 25, 30], [35, 40, 45]))

arr_2d[1, 0]
```

```
20
```

This grabs the value in row index 1, column index 0.

You can also slice out a rectangular section of a matrix, exactly like slicing a one dimensional array, just with two ranges instead of one.

```python
arr_2d[:2, 1:]
```

```
array([[10, 15],
       [25, 30]])
```

This reads as, take rows up to but not including index 2, and columns starting from index 1 onward.

### Splitting a Dataset Into Input Features (X) and Output (y)

This next pattern is one of the most important things to get comfortable with, since you will see it in almost every single machine learning project you ever work on. Real datasets are usually organized as a table, where the first several columns are the input features, and the very last column is the value you are trying to predict.

```python
arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

X = arr[:, :-1]    # all rows, all columns except the last one -> input features
y = arr[:, -1]     # all rows, only the last column -> output/target

print("Input data:\n", X)
print("Output data:\n", y)
```

```
Input data:
 [[1 2]
 [4 5]
 [7 8]]
Output data:
 [3 6 9]
```

Breaking down exactly what is happening inside those brackets removes any confusion. Remember the general format is `arr[rows, columns]`.

For `X = arr[:, :-1]`, the `:` before the comma means select all rows. The `:-1` after the comma means start from column 0 and stop right before the last column, so if there are 3 columns numbered 0, 1 and 2, this grabs columns 0 and 1 only.

For `y = arr[:, -1]`, the `:` again means all rows, while `-1` this time is a single index, not a range, so it grabs only the very last column by itself, using the negative indexing idea covered earlier.

A simple table view of the same data makes this even clearer.

```
        Col0  Col1  Col2
       +----+----+----+
Row0   |  1 |  2 |  3 |
       +----+----+----+
Row1   |  4 |  5 |  6 |
       +----+----+----+
Row2   |  7 |  8 |  9 |
       +----+----+----+
```

`X = arr[:, :-1]` keeps Col0 and Col1 for every row, and `y = arr[:, -1]` keeps only Col2 for every row.

This exact pattern is why you will see a line almost identical to `X = data[:, :-1]` and `y = data[:, -1]` at the very start of nearly every machine learning script you come across. Consider a small dataset of Age, Salary, Experience and Purchased.

```
Age   Salary   Experience   Purchased
25    40000    2            0
30    60000    5            1
35    80000    8            1
```

Here `X = data[:, :-1]` gives you Age, Salary and Experience as the input features for every row, and `y = data[:, -1]` gives you only the Purchased column, which is exactly the value the model is trying to learn to predict from the rest.

### Fancy Indexing

Fancy indexing lets you pick out several specific rows, in any order you like, by passing a list of index positions instead of a single number or a range.

```python
arr2d[[2, 4, 6, 8]]
```

This pulls out rows 2, 4, 6 and 8, in exactly that order. You can even rearrange the order freely.

```python
arr2d[[6, 4, 2, 7]]
```

This returns row 6 first, then row 4, then row 2, then row 7, showing that fancy indexing does not require the indices to be in increasing order.

### Selecting Values Using Conditions

One of the most powerful and frequently used NumPy features in machine learning is the ability to select values based on a condition, rather than a fixed position.

```python
arr = np.arange(1, 11)

arr > 4
```

```
array([False, False, False, False,  True,  True,  True,  True,  True, True])
```

Comparing an array to a number does not give you a single true or false answer, it gives you back a whole array of true and false values, one for every position, describing whether that condition held at that position. This is called a boolean array, and you can use it directly inside brackets to filter the original array down to only the values where the condition was true.

```python
arr[arr > 4]
```

```
array([ 5,  6,  7,  8,  9, 10])
```

This single line is doing an enormous amount of work behind the scenes, and it is exactly the kind of filtering you will use constantly when cleaning datasets, for example keeping only rows where a feature value is above a certain threshold.

## NumPy Operations

### Arithmetic

NumPy lets you perform arithmetic directly between two arrays of the same shape, applying the operation element by element, without writing a single loop yourself.

```python
arr = np.arange(0, 10)

arr + arr
arr * arr
arr - arr
```

Each of these operations combines the two arrays position by position, so the first element of the result comes from combining the first elements of both arrays, the second comes from combining the second elements, and so on.

There is one small but important behavior worth knowing about division. Dividing an array by itself where a zero is involved does not crash your program the way it might in plain Python. Instead, NumPy quietly gives you a warning and fills that particular position with `nan`, meaning not a number, and continues running.

```python
arr / arr
```

```
array([nan,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
```

Similarly, dividing a number by zero returns infinity instead of crashing.

```python
1 / arr
```

```
array([       inf, 1.        , 0.5       , 0.33333333, 0.25      ,
       0.2       , 0.16666667, 0.14285714, 0.125     , 0.11111111])
```

Knowing this behavior in advance saves a lot of confusion later, since your code will not throw an error, it will simply produce `nan` or `inf` values that you need to be aware of and handle appropriately.

Raising every element of an array to a power works exactly as you would expect too.

```python
arr ** 3
```

```
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
```

### Vector Dot Product

Multiplying two vectors with `*` gives you element-wise multiplication, as shown above, but this is not the only way two vectors can be combined. The dot product multiplies corresponding elements together and then adds up all of those products into a single number, and it is one of the most fundamental operations in all of machine learning, showing up everywhere from computing predictions in linear regression to computing similarity between two pieces of text.

```python
a = np.array([1, 2, 3])   # first vector
b = np.array([1, 2, 3])   # second vector

c = a.dot(b)               # compute the dot product
print(c)
```

```
14
```

This result comes from `1×1 + 2×2 + 3×3`, which is `1 + 4 + 9`, giving `14`. Notice the key difference from element-wise multiplication: `a * b` would have given you back a vector, `[1, 4, 9]`, while `a.dot(b)` collapses that same calculation down into one single scalar number.

### Vector Norms: Measuring Length With Code

The Linear Algebra note in this series already covered what a norm means mathematically. Here is how to actually compute one in NumPy, using the `norm` function from `numpy.linalg`.

```python
from numpy.linalg import norm

a = np.array([1, 2, 3])   # define a vector

l1 = norm(a, 1)            # calculate the L1 norm
print(l1)
```

```
6.0
```

The L1 norm simply adds up the absolute values of every entry, so `1 + 2 + 3` gives `6.0`. Leaving out the second argument calculates the L2 norm, also called the Euclidean norm, by default.

```python
l2 = norm(a)                # calculate the L2 norm (default)
print(l2)
```

```
3.7416573867739413
```

This matches `sqrt(1^2 + 2^2 + 3^2)`, which is `sqrt(14)`, confirming the L2 norm formula from the earlier linear algebra note works exactly the same way in code as it does on paper.

### Universal Array Functions

Beyond basic arithmetic, NumPy provides a large collection of mathematical functions, often called universal functions, that apply a mathematical operation across every element of an array at once.

```python
np.sqrt(arr)
```

This takes the square root of every single value in the array in one call.

```python
np.exp(arr)
```

This raises the mathematical constant e to the power of every value in the array, which shows up constantly in machine learning, especially inside functions like the sigmoid and softmax activation functions used in neural networks.

```python
np.sin(arr)
```

This applies the sine function element by element, useful whenever you are working with periodic or wave-like data.

```python
np.log(arr)
```

This applies the natural logarithm to every element. Note that taking the log of 0 produces negative infinity along with a warning, rather than crashing your program, following the same pattern as the division behavior mentioned earlier.

You can also apply many of these same operations directly as NumPy level functions rather than array methods, and they behave identically.

```python
np.max(arr)
```

This is exactly equivalent to writing `arr.max()`, just written as a function instead of a method.

## Matrices in NumPy

A matrix in NumPy is just a two dimensional `ndarray`, the same object type covered earlier, so everything already learned about arrays applies directly. This section covers the operations that are specific to working with matrices in a linear algebra sense.

Basic addition and subtraction between two matrices of the same shape work exactly the same way as they did for vectors earlier, combining corresponding positions.

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

C = A + B    # add matrices position by position
print(C)
```

```
[[ 2  4  6]
 [ 8 10 12]]
```

### Doubt Solved: Why Does A \* B Not Give Matrix Multiplication?

This is one of the most common points of confusion in NumPy, so it deserves a careful, dedicated explanation rather than a quick note. In plain linear algebra, when you write A times B, you almost always mean proper matrix multiplication. In NumPy code, however, the `*` operator does not do that at all. It performs element-wise multiplication instead, also called the Hadamard product.

```python
Mul = A * B    # this is NOT matrix multiplication
print(Mul)
```

```
[[ 1  4  9]
 [16 25 36]]
```

Here is exactly what happened. NumPy simply multiplied every position in `A` with the matching position in `B`.

```
A[0,0] × B[0,0] = 1 × 1 = 1
A[0,1] × B[0,1] = 2 × 2 = 4
A[0,2] × B[0,2] = 3 × 3 = 9
A[1,0] × B[1,0] = 4 × 4 = 16
A[1,1] × B[1,1] = 5 × 5 = 25
A[1,2] × B[1,2] = 6 × 6 = 36
```

This is only possible in the first place because `A` and `B` share the exact same shape, `(2, 3)`, so NumPy always knows exactly which position in `A` corresponds to which position in `B`.

Real matrix multiplication is a completely different calculation. Instead of multiplying matching positions, it combines entire rows of the first matrix with entire columns of the second matrix. To actually perform matrix multiplication in NumPy, use the `@` operator, or equivalently the `matmul` or `dot` functions.

```python
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])          # shape (2, 3)

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])          # shape (3, 2)

C = A @ B          # proper matrix multiplication
print(C)
```

```
[[ 58  64]
 [139 154]]
```

The first value, 58, comes from combining the entire first row of `A` with the entire first column of `B`, calculated as `1×7 + 2×9 + 3×11`, which is `7 + 18 + 33`, giving `58`. This is a genuinely different calculation from the Hadamard product shown above, and it also has a different requirement: proper matrix multiplication needs the number of columns in the first matrix to match the number of rows in the second matrix, which is why `A` needed shape `(2, 3)` and `B` needed shape `(3, 2)` here, unlike the Hadamard product, which needed both matrices to share the exact same shape.

Here is a quick reference table summarizing every operator involved.

| Operation         | Symbol   | Meaning                                                            |
| ----------------- | -------- | ------------------------------------------------------------------ |
| `A + B`           | `+`      | Element-wise addition                                              |
| `A - B`           | `-`      | Element-wise subtraction                                           |
| `A * B`           | `*`      | Element-wise multiplication (Hadamard product)                     |
| `A / B`           | `/`      | Element-wise division                                              |
| `A @ B`           | `@`      | Proper matrix multiplication                                       |
| `np.matmul(A, B)` | function | Proper matrix multiplication                                       |
| `np.dot(A, B)`    | function | Matrix multiplication for 2D arrays, and dot product for 1D arrays |

### Types of Matrices

A few specific matrix shapes come up repeatedly enough in machine learning and linear algebra that NumPy provides direct functions for building them.

A diagonal matrix has nonzero values only along its diagonal, with everything else set to zero. You can pull the diagonal values out of an existing matrix, or build a brand new diagonal matrix from a list of values, using the same `diag` function.

```python
from numpy import diag

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

d = diag(M)     # extract the diagonal values from M
print(d)

D = diag(d)      # build a new diagonal matrix from those values
print(D)
```

```
[1 5 9]
[[1 0 0]
 [0 5 0]
 [0 0 9]]
```

A triangular matrix has all its values above or below the diagonal set to zero, depending on which triangle you keep.

```python
from numpy import tril, triu

M = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
])

lower = tril(M)   # keep only the lower triangle, zero out the rest
upper = triu(M)   # keep only the upper triangle, zero out the rest

print(lower)
print(upper)
```

```
[[1 0 0]
 [1 2 0]
 [1 2 3]]
[[1 2 3]
 [0 2 3]
 [0 0 3]]
```

An identity matrix has 1s along the diagonal and 0s everywhere else, and behaves in matrix multiplication the same way the number 1 behaves in ordinary multiplication, meaning multiplying any matrix by an identity matrix of the right shape leaves it unchanged.

```python
from numpy import identity

I = identity(3)   # create a 3 by 3 identity matrix
print(I)
```

```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

### Transpose

The transpose of a matrix flips it over its diagonal, turning every row into a column and every column into a row. You already saw this idea in the matrix calculus note, and in NumPy it is available directly as the `.T` attribute.

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

C = A.T    # transpose A, flipping rows into columns
print(C)
```

```
[[1 3 5]
 [2 4 6]]
```

### Inverse

The inverse of a square matrix is another matrix that, when multiplied with the original, gives back the identity matrix, playing a role similar to how dividing by a number undoes multiplying by that number. Not every matrix has an inverse, but when one exists, NumPy can compute it directly.

```python
from numpy.linalg import inv

A = np.array([
    [1.0, 2.0],
    [3.0, 4.0]
])

B = inv(A)    # compute the inverse of A
print(B)
```

```
[[-2.   1. ]
 [ 1.5 -0.5]]
```

### Determinant

The determinant is a single scalar number calculated from a square matrix, and it carries a geometric meaning, roughly describing how much a transformation represented by that matrix stretches or shrinks space. A determinant of zero specifically tells you the matrix cannot be inverted, which is a very useful early warning sign when working with linear systems.

```python
n_array = np.array([
    [50, 29],
    [30, 44]
])

det = np.linalg.det(n_array)   # calculate the determinant
print(int(det))
```

```
1330
```

## Summary Table

| Function or Concept                             | Purpose                                                                       |
| ----------------------------------------------- | ----------------------------------------------------------------------------- |
| `np.array()`                                    | Convert a Python list into a NumPy array                                      |
| `type(a)` -> `numpy.ndarray`                    | The fixed-size, single-datatype array object underlying every NumPy array     |
| `np.vstack()` / `np.hstack()`                   | Combine arrays vertically (new rows) or horizontally (new columns)            |
| `np.arange()`                                   | Generate evenly spaced values using a step size                               |
| `np.zeros()` / `np.ones()`                      | Generate arrays filled with 0s or 1s                                          |
| `np.linspace()`                                 | Generate a fixed number of evenly spaced values                               |
| `np.eye()`                                      | Generate an identity matrix                                                   |
| `np.random.rand()`                              | Generate uniformly distributed random values                                  |
| `np.random.randn()`                             | Generate normally distributed random values                                   |
| `np.random.randint()`                           | Generate random integers within a range                                       |
| `.reshape()`                                    | Rearrange an array into a new shape                                           |
| `.max()` / `.min()`                             | Find the largest or smallest value                                            |
| `.argmax()` / `.argmin()`                       | Find the index of the largest or smallest value                               |
| `.shape`                                        | Check the dimensions of an array                                              |
| `.dtype`                                        | Check the data type stored in an array                                        |
| Slicing `arr[a:b]`                              | Select a range of elements, returns a view not a copy                         |
| Negative slicing `arr[-2:]`                     | Select elements counting from the end of the array                            |
| `arr[:, :-1]` and `arr[:, -1]`                  | The standard pattern for splitting a dataset into features (X) and target (y) |
| `.copy()`                                       | Create an independent copy of an array                                        |
| Fancy indexing `arr[[i, j, k]]`                 | Select specific rows in any order                                             |
| Boolean indexing `arr[arr > x]`                 | Select elements matching a condition                                          |
| Broadcasting                                    | Combine arrays of different shapes, such as an array and a single scalar      |
| Arithmetic (`+`, `-`, `*`, `/`, `**`)           | Perform element-wise operations between arrays                                |
| `.dot()` (1D arrays)                            | Compute the vector dot product, a single scalar                               |
| `numpy.linalg.norm()`                           | Compute the L1, L2, or other norm of a vector                                 |
| `np.sqrt()`, `np.exp()`, `np.sin()`, `np.log()` | Apply a mathematical function across every element                            |
| `A * B` (2D arrays)                             | Element-wise multiplication of matrices, also called the Hadamard product     |
| `A @ B`, `np.matmul()`, `np.dot()` (2D arrays)  | Proper matrix multiplication                                                  |
| `diag()`                                        | Extract a diagonal from a matrix, or build a diagonal matrix from a vector    |
| `tril()` / `triu()`                             | Keep only the lower or upper triangle of a matrix                             |
| `identity()`                                    | Generate an identity matrix                                                   |
| `.T`                                            | Transpose a matrix, swapping rows and columns                                 |
| `numpy.linalg.inv()`                            | Compute the inverse of a square matrix                                        |
| `numpy.linalg.det()`                            | Compute the determinant of a square matrix                                    |

## Final Thoughts

NumPy might seem like just another library at first glance, but it is really the foundation everything else in the Python machine learning world sits on top of. The moment you load a dataset with Pandas, train a model with Scikit-learn, or build a neural network with TensorFlow or PyTorch, NumPy arrays, or something modeled directly after them, are working quietly underneath.

Getting genuinely comfortable with creating arrays, reshaping them, indexing into them, and applying operations across them is time well spent, since almost nothing else in machine learning code makes sense until this feels natural. Practice by taking a small dataset of your own, loading it as a NumPy array, and trying to reshape it, filter it, and run a few universal functions on it. That hands on repetition is what turns this from a reference guide into a skill you actually own.
