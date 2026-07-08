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

## Two Ways to Write Matplotlib Code

Before going further, there is one thing that confuses almost every beginner at some point, so let us clear it up right now. You will see matplotlib code written in two different styles, and both are completely correct.

### Style 1: Just use plt.plot() directly

Most tutorials, including many beginner notes, start here. You simply call `plt.plot()` and matplotlib quietly creates a figure and axes for you in the background, without you ever asking for them by name.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.array([20, 30, 40, 50])
y = np.array([3, 2, 5, 4])

plt.plot(x, y)
plt.xlabel("Test Number")
plt.ylabel("Test Score")
plt.show()
```

This is called the pyplot style, or the state based interface. Matplotlib always keeps track of "the current figure" and "the current axes" internally, and every `plt.something()` call simply acts on whatever is currently active. You do not need to create `fig` or `ax` at all. This is quick to write and perfectly fine for a single, simple plot, which is exactly why it is taught first almost everywhere.

### Style 2: Create fig and ax yourself

The second style is where you explicitly create the figure and the axes with your own variable names, and then call methods on them directly.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(x, y)
ax.set_xlabel("Test Number")
ax.set_ylabel("Test Score")
ax.set_title("Test Performance")

plt.show()
```

This is called the object oriented style. You do not need it for a simple single plot, but it becomes very useful the moment you want multiple charts inside one figure, or you want very precise control over layout. This is also the style you will see more often in real projects, research code, and production dashboards.

### So which one should you use

If you are comfortable with `plt.plot()` directly, keep using it. It works correctly, it is not outdated, and it is the right choice for most simple plots. Nothing in this guide requires you to switch. Just know that whenever you see `fig` or `ax` in code, including further down in this same guide, it is only the second style being used, doing the exact same job in a more structured way. Think of it as two paths to the same destination, not two different destinations.

## The Building Blocks: Figure, Axes, Axis, Labels and Ticks

This is the part most beginners skip, and it is the exact part that causes confusion later. Understanding these five words properly will make everything else click. These names describe the object oriented style mentioned above, so even if you always write `plt.plot()` directly, matplotlib is still creating one figure and one axes for you behind the scenes. Knowing what those words mean will help you understand code you come across later, even if you keep writing `plt.plot()` in your own work.

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

### Customizing Labels and Title with a Font Dictionary

If you want your labels and title to look a certain way, matplotlib lets you pass a small dictionary describing the font, instead of setting each property one by one.

```python
font = {
    "family": "Arial",
    "color": "navy",
    "size": 18
}

plt.title("Test Performance", fontdict=font)
plt.xlabel("Test Number", fontdict=font)
plt.ylabel("Test Score", fontdict=font)
```

The dictionary can hold three common keys. `family` sets the font family, `color` sets the text color, and `size` sets the font size. Once you define this dictionary, you can reuse the same `font` variable for the title and both axis labels, so all the text in your chart looks consistent.

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

Calling `plt.grid()` with no arguments at all displays both the horizontal and the vertical grid lines together.

### Showing Grid Lines on Only One Axis

Sometimes you only want vertical lines, or only horizontal lines, instead of both. The `axis` keyword controls this.

```python
plt.grid(axis="x")
```

This shows only the vertical grid lines, drawn at the x axis tick positions.

```python
plt.grid(axis="y")
```

This shows only the horizontal grid lines, drawn at the y axis tick positions.

### Different Styling for Each Axis

Since `plt.grid()` can be called more than once, you can style the vertical and horizontal grid lines differently in the same chart.

```python
plt.grid(axis="y", color="green")
plt.grid(axis="x", color="red")
```

The same idea works for line style and line width as well, using `linestyle` or `ls`, and `linewidth` or `lw`.

```python
plt.grid(ls="dotted")
plt.grid(lw=3)
```

## Subplots: Multiple Graphs in One Figure

So far every example has drawn a single chart. Often in machine learning work, you want to place several related charts side by side, for example comparing training loss and validation accuracy in one glance, or comparing the same metric across several models.

`plt.subplot()` lets you divide one figure into a grid of smaller plotting areas, and then draw into each one separately.

```python
plt.subplot(rows, cols, index)
```

Here `rows` and `cols` describe the shape of the grid, and `index` tells matplotlib which cell in that grid you are about to draw into. The index always starts counting from 1, not 0, and it fills the grid left to right, then top to bottom.

### A Simple Side by Side Example

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y1 = [10, 20, 25, 30]
y2 = [5, 15, 10, 25]

plt.subplot(1, 2, 1)
plt.plot(x, y1)
plt.title("Model A")

plt.subplot(1, 2, 2)
plt.plot(x, y2)
plt.title("Model B")

plt.show()
```

This creates one row with two columns, so the two charts appear side by side. The first `plt.subplot(1, 2, 1)` fills the first cell, and `plt.subplot(1, 2, 2)` fills the second cell.

### A Bigger Grid

The same idea extends to any grid size. A 2 by 3 grid, for example, is filled like this.

```python
plt.subplot(2, 3, 1)
plt.subplot(2, 3, 2)
plt.subplot(2, 3, 3)
plt.subplot(2, 3, 4)
plt.subplot(2, 3, 5)
plt.subplot(2, 3, 6)
```

The positions are filled in this order.

```
1  2  3
4  5  6
```

### Titles for Subplots

Each individual subplot can have its own title using `plt.title()` right after that subplot is created, exactly like the single chart examples earlier in this guide. If you also want one title that applies to the entire figure, above all the subplots, use `plt.suptitle()` instead.

```python
plt.suptitle("Model Comparison Report")
```

`plt.title()` labels one specific chart. `plt.suptitle()` labels the whole figure that contains all of them.

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

### What Happens if You Only Give One Array

You do not always need to pass both x and y values. If you give `plt.plot()` only one array, matplotlib treats it as the y values, and automatically fills in the x values as 0, 1, 2, 3 and so on.

```python
y = np.array([3, 2, 5, 4])

plt.plot(y)
plt.show()
```

Here the points plotted are effectively (0, 3), (1, 2), (2, 5) and (3, 4). Note that this only works in one direction. If you pass a single array intending it to be x values, matplotlib will still treat it as y values, since there is no way for it to know your intention otherwise.

### Showing Only Points, No Line

Sometimes you do not want a connecting line at all, just the individual points. Add `'o'` as a third argument to switch to points only.

```python
plt.plot(x, y, 'o')
plt.show()
```

### Understanding Markers

A marker is simply the symbol matplotlib draws at each data point. You have already seen `'o'` for a circle. There are many more shapes available.

| Marker | Shape |
|---|---|
| `'o'` | Circle |
| `'x'` | Small X |
| `'X'` | Bold X |
| `'*'` | Star |
| `'D'` | Diamond |
| `'^'` | Triangle Up |
| `'v'` | Triangle Down |
| `'s'` | Square |
| `'+'` | Plus |

There are two ways to add a marker. Writing it as a short string like `plt.plot(x, y, 'o')` shows only the markers with no line. Writing it using the keyword `marker="o"` shows both the markers and the connecting line together.

```python
plt.plot(x, y, marker="D")
plt.show()
```

### Combining Marker, Line Style and Color in One String

Matplotlib lets you describe the marker, the line style and the color together in a single short format string, in that exact order.

```python
plt.plot(x, y, 'o:r')
plt.show()
```

Here `o` is a circle marker, `:` is a dotted line, and `r` is the color red. Another example, `'o-g'`, means a circle marker, a solid line, and green color.

### Line Style

Line style controls how the connecting line looks. You can set it using the `linestyle` keyword, or its shorter form `ls`.

```python
plt.plot(x, y, ls="solid")
plt.plot(x, y, ls="dashed")
plt.plot(x, y, ls="dotted")
plt.plot(x, y, ls="dashdot")
plt.plot(x, y, ls="none")
```

`"none"` means no connecting line is drawn at all, which is useful if you only want to display markers using the `marker` keyword. The same styles can also be written using symbols instead of words, where `-` means solid, `--` means dashed, `:` means dotted, and `-.` means dash dot.

### Line Color

You can set the color of the line using the `color` keyword, and you can specify that color in three different ways.

```python
plt.plot(x, y, color="red")
plt.plot(x, y, color="#AA00AA")
```

The first way is a short color code, where `r` is red, `g` is green, `b` is blue, `c` is cyan, `m` is magenta, `y` is yellow, `k` is black, and `w` is white. The second way is a full color name, such as `"purple"`, `"orange"` or `"brown"`, since matplotlib understands a large number of standard color names. The third way is a hexadecimal color code, such as `"#FF0000"` for red or `"#00FF00"` for green, which gives you access to virtually any shade you want.

### Line Width

The `linewidth` keyword, or its shorter form `lw`, controls how thick the plotted line appears.

```python
plt.plot(x, y, lw=5)
```

### Marker Size and Marker Colors

Beyond the marker shape, you can also control its size and its coloring in detail.

```python
plt.plot(
    x,
    y,
    marker="o",
    ms=15,
    mec="black",
    mfc="yellow"
)
```

`markersize`, shortened to `ms`, controls how big each marker is. `markeredgecolor`, shortened to `mec`, controls the border color of the marker. `markerfacecolor`, shortened to `mfc`, controls the fill color inside the marker. All three color values can be given as a color code, a color name, or a hex code, exactly like the line color.

### Plotting Multiple Lines on the Same Graph

You can call `plt.plot()` more than once before calling `plt.show()`, and every call adds one more line to the same chart. Each line can have its own color, marker, line style and width, which makes this the standard way to compare things like training loss and validation loss on one graph.

```python
plt.plot(x1, y1, color="blue", label="Training Loss")
plt.plot(x2, y2, color="orange", label="Validation Loss")
plt.legend()
plt.show()
```

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
