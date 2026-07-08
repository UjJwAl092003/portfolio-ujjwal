---
title: Matplotlib for Machine Learning: A Complete Beginner's Guide
slug: matplotlib-for-machine-learning-a-complete-beginners-guide
date: "2026-07-05"
category: Machine Learning
readingTime: "12 min read"
description: "If you are starting your machine learning journey, you will hear one thing again and again: data alone does not tell a story. You have to see it to understand it. That is exactly why Matplotlib exists."
tags:
  - "Machine Learning"
  - "Matplotlib"
  - "Data Visualization"
  - "Python"
featuredImage: "/images/blogs/matplotlib.png"
author: "Ujjwal Singh"
---

# Matplotlib for Machine Learning: A Complete Beginner's Guide

If you are starting your machine learning journey, you will hear one thing again and again. Data alone does not tell a story. You have to see it to understand it. That is exactly why Matplotlib exists.

This guide is written for someone who has never touched Matplotlib before. By the end, you will not just know the syntax. You will know why each plot exists, when to use it, and how to read it like a machine learning practitioner does.

Let us begin from zero.

## What is Matplotlib and why does it matter in ML

Matplotlib is a plotting library for Python. It lets you turn rows and columns of numbers into pictures that your brain can actually process. A table with ten thousand rows means nothing to your eyes. A graph of the same ten thousand rows can show you a trend in two seconds.

In machine learning, you constantly need to answer questions like these. Is this feature normally distributed. Are these two variables related. Is my model improving over each training step. Does my data have outliers. Matplotlib is the tool that answers all of these visually.

It has been around for a long time, and it was designed to feel familiar to people coming from MATLAB. That is why a lot of its function names and style will look similar if you have used MATLAB before.

## Installing Matplotlib

Before plotting anything, you need the library installed. Pick whichever matches your setup.

```
pip install matplotlib
```

If you are using Anaconda, use this instead.

```
conda install matplotlib
```

## Importing Matplotlib

Almost everything you do in Matplotlib comes from one module called pyplot. The community convention is to import it like this.

```python
import matplotlib.pyplot as plt
```

You will see `plt` used everywhere from here on. Just remember, `plt` is simply a short name for `matplotlib.pyplot`.

## The Building Blocks: Figure, Axes, Axis, Labels and Ticks

This is the part most beginners skip, and it is the exact part that causes confusion later. Understanding these five words properly will make everything else click.

### Figure

Think of the Figure as the entire canvas or the whole window on which your plot will live. One figure can hold one plot, or it can hold many plots side by side.

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4), facecolor="lightyellow", edgecolor="black", dpi=100)
```

Here `figsize` controls the width and height of the canvas in inches, and `dpi` controls how sharp or detailed the image is.

### Axes

If Figure is the canvas, Axes is the actual drawing area inside it, the part where your data, your x axis, your y axis and your title all live. One figure can contain several axes, which is how you create multiple charts in a single image.

```python
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlabel("Feature Value")
ax.set_ylabel("Frequency")
ax.set_title("Understanding Axes")
```

A simple rule to remember. Figure is the frame on the wall. Axes is the painting inside that frame.

### Axis

Axis refers to the actual number line, the horizontal or vertical reference line along which your values are measured. You can control how much of that line is visible using limits.

```python
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
```

### Labels

Labels are simply the names you give to your x axis, y axis, or the whole chart, so that anyone looking at the plot understands what it represents without you having to explain it.

```python
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
```

### Ticks

Ticks are the small markers along the axis that show specific points, like the numbers you see along a ruler. You can decide exactly where they appear and what text label goes next to each one.

```python
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri"])
```

This is extremely useful in ML when your x axis represents something like epoch numbers or days, and you want readable names instead of plain numbers.

## Legend

When you plot more than one line or one category on the same chart, a legend tells the reader which color or line belongs to which thing.

```python
import numpy as np

x = np.arange(0, 10, 0.1)
plt.plot(x, np.sin(x), color="blue")
plt.plot(x, np.cos(x), color="orange")
plt.legend(["sine wave", "cosine wave"], loc="upper right")
plt.show()
```

The `loc` parameter decides where the legend box sits. Common values are upper right, upper left, lower right, lower left, and best, where best lets Matplotlib automatically pick the least cluttered spot.

## Grid Lines

Grid lines are the faint lines running across the plot area that make it easier to estimate values at a glance, similar to the lines on graph paper.

```python
plt.grid(color="grey", linestyle="--", linewidth=0.5)
```

## A Few Functions You Will Use Constantly

Before jumping into plot types, memorize these small but important functions.

`plt.show()` displays the figure on your screen.

`plt.savefig("myplot.png", dpi=300, bbox_inches="tight")` saves your figure as an image file, which is useful when you want to include the chart in a report or a blog like this one.

`plt.title()` sets the title text of your chart.

`plt.annotate()` lets you add text, arrows or notes pointing at specific parts of your chart, which is great for highlighting an outlier or an important point.

Now that the foundation is clear, let us go through the plots that actually matter for machine learning work.

## Line Plot

A line plot connects data points with straight lines. It is the natural choice whenever you are tracking something over time or over a sequence, for example the loss value of a neural network across training epochs.

```python
import matplotlib.pyplot as plt

epochs = list(range(1, 11))
training_loss = [2.3, 1.9, 1.5, 1.2, 1.0, 0.85, 0.7, 0.6, 0.55, 0.5]

plt.plot(epochs, training_loss, color="black")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
```

In machine learning, this exact plot is what you see when you track loss or accuracy while a model trains.

## Area Plot

An area plot is a line plot where the space between the line and the axis is filled with color. It helps emphasize magnitude rather than just direction, which makes it useful when comparing total volumes across categories, like monthly resource usage.

```python
import numpy as np

months = np.arange(12)
usage = [20, 25, 22, 30, 35, 40, 45, 42, 38, 33, 28, 24]

plt.fill_between(months, usage, color="skyblue", alpha=0.5)
plt.plot(months, usage, color="black")
plt.xlabel("Month")
plt.ylabel("Resource Usage")
plt.show()
```

### Stacked Area Plot

When you have multiple categories and you want to see both their individual contribution and their combined total at the same time, a stacked area plot is the right choice. Each category sits on top of the previous one.

```python
x = range(1, 6)
category_a = [1, 4, 6, 8, 9]
category_b = [2, 2, 7, 10, 12]
category_c = [2, 8, 5, 10, 6]

plt.stackplot(x, category_a, category_b, category_c, labels=["A", "B", "C"])
plt.legend(loc="upper left")
plt.show()
```

## Scatter Plot

A scatter plot places a single dot for every pair of x and y values. It is one of the most important plots in machine learning because it instantly reveals whether two variables are related, and it is the go to way of checking correlation before building a model.

```python
import numpy as np

np.random.seed(42)
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, color="black", alpha=0.5)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

### Bubble Plot

A bubble plot is a scatter plot with a third dimension added through the size of each dot. Bigger dots represent bigger values of that third variable.

```python
area = (30 * np.random.rand(50)) ** 2

plt.scatter(x, y, s=area, c="black", alpha=0.5)
plt.show()
```

You can even add a fourth dimension by coloring each dot differently based on another variable.

```python
colors = np.random.rand(50)
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
```

## Bar Plot

A bar plot uses rectangular bars to represent values across categories, and it is the standard choice whenever you are comparing distinct groups rather than a continuous trend.

```python
languages = ["Python", "C++", "Java", "R"]
learners = [9000, 6300, 10000, 4000]

plt.bar(languages, learners, color="black", alpha=0.7)
plt.xlabel("Programming Language")
plt.ylabel("Number of Learners")
plt.show()
```

### Grouped Bar Plot

When you want to compare multiple series across the same categories, bars are placed next to each other in groups instead of overlapping.

```python
import numpy as np

matches = ["Match 1", "Match 2", "Match 3"]
player_a = [30, 25, 50]
player_b = [40, 23, 51]

x_pos = np.arange(len(matches))
plt.bar(x_pos - 0.2, player_a, width=0.4, label="Player A")
plt.bar(x_pos + 0.2, player_b, width=0.4, label="Player B")
plt.xticks(x_pos, matches)
plt.legend()
plt.show()
```

### Stacked Bar Plot

Here, instead of placing bars side by side, one is drawn on top of another, which is helpful for showing how a total is made up of smaller parts.

```python
groups = ["G1", "G2", "G3"]
male = [20, 35, 30]
female = [25, 32, 34]

plt.bar(groups, male, color="black", label="Male")
plt.bar(groups, female, bottom=male, color="orange", label="Female")
plt.legend()
plt.show()
```

## Histogram

A histogram is arguably the single most used plot in machine learning. It shows how your data is distributed across value ranges, which is essential before doing anything with a feature, since the shape of the distribution can tell you if a feature is skewed, normal, or has strange gaps.

```python
import numpy as np

marks = np.array([22, 87, 5, 43, 56, 73, 55, 54, 11, 20, 51, 5, 79, 31, 27])

plt.hist(marks, bins=[0, 20, 40, 60, 80, 100], color="green", alpha=0.5, edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Distribution of Marks")
plt.show()
```

You can also overlay two histograms to compare two distributions directly, which is very common when comparing training data against test data.

## Density Plot

A density plot is a smoothed version of a histogram. Instead of bars, it draws a continuous curve estimating the probability distribution of your data, which makes patterns easier to see when your dataset is large.

```python
import pandas as pd

data = pd.DataFrame(marks)
data.plot(kind="density", color="red", alpha=0.6)
plt.show()
```

## Pie Plot

A pie plot shows how a whole is divided into parts, where each slice represents a percentage of the total. It is best used when you have a small number of categories and you care about relative proportion rather than exact values.

```python
labels = ["English", "Spanish", "Russian", "French", "Chinese"]
sizes = [55, 15, 8, 10, 12]

plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.axis("equal")
plt.show()
```

### Nested Pie Plot

A nested pie plot shows two levels of categories at once, an inner ring and an outer ring, which is useful for showing a category and its sub categories together.

## Box Plot

A box plot is one of the most important tools for exploratory data analysis in machine learning. In a single compact shape, it shows you the minimum, the maximum, the median, and the spread of your data through quartiles. It is the fastest way to spot outliers.

```python
import numpy as np

values = np.random.normal(100, 10, 200)

plt.boxplot(values)
plt.title("Box Plot of Feature Values")
plt.show()
```

Reading a box plot. The box in the middle represents the middle fifty percent of your data. The line inside the box is the median. The lines extending outward are called whiskers, and any dots beyond the whiskers are potential outliers.

### Multiple Box Plots

Placing several box plots side by side lets you compare the spread and outliers of multiple features or multiple groups in one glance.

```python
course1 = [82, 76, 24, 40, 67, 62, 75, 78, 71, 32]
course2 = [62, 5, 91, 25, 36, 32, 96, 95, 3, 90]

plt.boxplot([course1, course2], labels=["Course 1", "Course 2"])
plt.show()
```

## Violin Plot

A violin plot combines a box plot with a density plot. Along with showing the median and the spread, it also shows you the shape of the distribution, almost like the box plot got fattened out at the points where more data exists.

```python
import numpy as np

data = np.random.normal(100, 10, 200)

plt.violinplot(data)
plt.title("Violin Plot")
plt.show()
```

Use a violin plot when a simple box plot is not enough and you also want to see whether your data has one peak, two peaks, or an unusual shape.

## Marginal Plot

A marginal plot is a scatter plot in the center with a histogram or box plot placed along the top and the side edges. This lets you see the relationship between two variables and their individual distributions all in one figure, which is extremely handy during exploratory data analysis when you are studying two features together.

## Stem Plot

A stem plot displays each data point as a small marker connected to the baseline by a straight line, similar to a lollipop shape. It works well for small datasets where you want to emphasize each individual value rather than a connected trend.

```python
data = [16, 25, 47, 56, 23, 45, 19, 55, 44, 27]
positions = list(range(1, 11))

plt.stem(positions, data)
plt.show()
```

## Step Plot

A step plot connects points using horizontal and vertical segments instead of a diagonal line, creating a staircase look. This is useful when a value stays constant for a while and then jumps suddenly, such as tracking a discrete setting that changes only at specific moments.

```python
years = [2018, 2019, 2020, 2021, 2022]
salary = [30000, 32000, 32000, 35000, 40000]

plt.step(years, salary, color="red")
plt.xlabel("Year")
plt.ylabel("Salary")
plt.show()
```

## Contour Plot

A contour plot represents a three dimensional surface using a two dimensional plane, where one axis is one variable, the second axis is another variable, and color or contour lines represent the third variable. This is very useful in machine learning when visualizing something like a loss surface or a cost function across two parameters.

```python
import numpy as np

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
x, y = np.meshgrid(x, y)
z = np.sin(x) + np.cos(y)

plt.contourf(x, y, z)
plt.colorbar()
plt.show()
```

## Heatmap

A heatmap represents a matrix of numbers using color intensity instead of raw numbers. In machine learning, heatmaps are used constantly to visualize things like correlation matrices between features, or confusion matrices after evaluating a classification model.

```python
import numpy as np

matrix = np.random.rand(5, 5)

plt.imshow(matrix, cmap="Blues")
plt.colorbar()
plt.show()
```

## Time Series Plot

A time series plot is simply a line plot where the x axis represents time, such as days, months or years, and the y axis represents the measured value. This is the standard visualization for anything sequential, like stock prices, sensor readings, or model performance logged over time.

```python
import pandas as pd

dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
values = [10, 12, 9, 14, 13, 17, 16, 19, 18, 21]

plt.plot(dates, values, color="red")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Example")
plt.show()
```

## Three Dimensional Plotting

Matplotlib also supports three dimensional plots through a toolkit called mplot3d. This is useful when you genuinely need to visualize three variables at once, such as visualizing a decision boundary in three dimensional feature space.

```python
from mpl_toolkits import mplot3d
import numpy as np

z = np.linspace(0, 1, 100)
x = z * np.cos(20 * z)
y = z * np.sin(20 * z)

ax = plt.axes(projection="3d")
ax.plot3D(x, y, z, "darkorange")
plt.show()
```

## Putting It All Together: Which Plot Should You Use

Here is a simple way to decide which plot fits your situation.

If you are tracking something over time or sequence, use a line plot or a time series plot.

If you want to check the relationship between two numeric variables, use a scatter plot.

If you want to compare values across categories, use a bar plot.

If you want to understand the distribution and spread of a single variable, use a histogram, a density plot, or a box plot.

If you want to spot outliers quickly, use a box plot or a violin plot.

If you want to see proportions of a whole, use a pie plot.

If you want to visualize a matrix of relationships, such as correlation between features, use a heatmap.

If you are working with three variables at once, use a contour plot or a three dimensional plot.

## Final Thoughts

Matplotlib might look overwhelming at first because of the sheer number of plot types and parameters, but almost everything builds on the same small set of ideas. You have a figure. Inside it you have axes. Inside the axes you plot your data, label your axes, add a legend if needed, and show or save the result.

Once you get comfortable with these building blocks, learning a new plot type becomes a matter of remembering one new function name rather than learning something entirely new. Start by practicing with your own small datasets, plot them in different ways, and slowly you will develop an instinct for which chart tells the story of your data best.

That instinct is exactly what separates someone who just runs machine learning code from someone who truly understands their data.
