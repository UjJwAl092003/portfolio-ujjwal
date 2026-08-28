# Notes: NumPy, Pandas, and Matplotlib Practice on a Sales Dataset

A question-by-question breakdown of this project's analysis, mapped directly from the notebook (`notebook.ipynb`). Written as a personal revision reference while learning **NumPy, Pandas, Matplotlib, and EDA** for machine learning.

The dataset used: a Sales dataset with columns `Region`, `Country`, `Item Type`, `Sales Channel`, `Order Priority`, `Order Date`, `Ship Date`, `Order ID`, `Units Sold`, `Unit Price`, `Unit Cost`, `Total Revenue`, `Total Cost`, `Total Profit` (100 rows total). See the main `README.md` for where to download it.

The goal here isn't deep theory, it's making sure I actually know **what function to reach for and why**, since that's what matters for practical ML work. Each question below shows my code and its real output, exactly as it ran. Where I skipped something or made a small mistake, I've marked it clearly so future-me doesn't repeat it.

---

## Before the sales dataset: general warm-up

Before starting the actual question list, I warmed up on a separate small dataset (`data.csv`, with `Duration`, `Pulse`, `Calories` columns) and on plain Matplotlib syntax with dummy numbers. This wasn't part of the official question list, but it's where I practiced things like:

- Checking for missing values: `df.isnull().sum()`
- Changing a column's type: `df['Pulse_float'] = df['Pulse'].astype('float')`
- Filling missing values with the median: `df['Calories_New'] = df['Calories'].fillna(df['Calories'].median())`
- Applying a custom function to a column: `df['Calories_New'].apply(square)`
- Basic `groupby()` with `.agg(['sum','mean','count'])`
- Merging two small DataFrames on a shared key column
- Basic Matplotlib: `plt.plot()`, styled lines (color, linestyle, marker), `plt.subplot()` grids, `plt.bar()`, `plt.hist()`, `plt.scatter()`, `plt.pie()`

Good fundamentals to have warmed up on, all of it reappears below on the real dataset, so I won't repeat the syntax explanations twice.

---

## 🟢 Level 1 — Dataset Understanding

**Q1. How many rows and columns are there?**

```python
sales.shape
```
Output: `(100, 14)` → 100 rows, 14 columns.

**Q2. What are all the column names?**

```python
sales.columns
```
Output:
```
Index(['Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority',
       'Order Date', 'Order ID', 'Ship Date', 'Units Sold', 'Unit Price',
       'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit'],
      dtype='object')
```

**Q3. What are the data types of every column?**

```python
sales.info()
```
Output:
```
RangeIndex: 100 entries, 0 to 99
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Region          100 non-null    str    
 1   Country         100 non-null    str    
 2   Item Type       100 non-null    str    
 3   Sales Channel   100 non-null    str    
 4   Order Priority  100 non-null    str    
 5   Order Date      100 non-null    str    
 6   Order ID        100 non-null    int64  
 7   Ship Date       100 non-null    str    
 8   Units Sold      100 non-null    int64  
 9   Unit Price      100 non-null    float64
 10  Unit Cost       100 non-null    float64
 11  Total Revenue   100 non-null    float64
 12  Total Cost      100 non-null    float64
 13  Total Profit    100 non-null    float64
dtypes: float64(5), int64(2), str(7)
memory usage: 16.9 KB
```
Handy detail: `.info()` alone answers Q1, Q3, Q6, *and* Q9 all at once (shape, dtypes, non-null counts, and memory usage), worth remembering as a one-stop command.

**Q4 & Q5. Which columns are numerical / categorical?**

```python
nums_cols = sales.select_dtypes(include='number').columns
cat_cols = sales.select_dtypes(include='str').columns
```
Output (categorical):
```
Index(['Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority',
       'Order Date', 'Ship Date'], dtype='str')
```
Output (numerical):
```
Index(['Order ID', 'Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue',
       'Total Cost', 'Total Profit'], dtype='str')
```
Note: `Order ID` shows up as numerical by dtype, but it's really an identifier, not a quantity you'd average or plot a histogram of. Worth remembering that "numerical dtype" and "numerically meaningful" aren't always the same thing.

**Q6. Are there any missing values?**

```python
sales.isnull().sum()
```
Output: every single column returned `0`. No missing values anywhere in this dataset.

**Q7. Are there any duplicate rows?**

```python
sales.duplicated().sum()
```
Output: `0` — no duplicate rows.

**Q8. Display 10 random rows.**

```python
sales.sample(10)
```
This returns 10 randomly chosen rows (not the same 10 every time you run it, since no `random_state` was fixed).

**Q9. What is the memory usage of the DataFrame?**

I actually used `sales.size` here, which gives `1400`. That's a mistake worth flagging for revision: **`.size` gives rows × columns (a count of cells), not memory usage.** The correct command is:

```python
sales.memory_usage(deep=True).sum()
```
or simply read it off the bottom of `sales.info()`, which already printed `memory usage: 16.9 KB`. Good catch for next time, `.size` and "memory usage" sound related but measure completely different things.

**Q10. Number of unique values in every categorical column.**

```python
sales.select_dtypes(include='str').nunique()
```
Output:
```
Region              7
Country            76
Item Type          12
Sales Channel       2
Order Priority      4
Order Date        100
Ship Date          99
```

---

## 🟡 Level 2 — Pandas Practice

**Q11. How many different countries?**

```python
sales['Country'].nunique()
```
Output: `76`.

**Q12. Orders per Region.**

```python
sales.groupby('Region')['Order ID'].count()
```
Output:
```
Asia                                 11
Australia and Oceania                11
Central America and the Caribbean     7
Europe                               22
Middle East and North Africa         10
North America                         3
Sub-Saharan Africa                   36
```

**Q13. Orders per Item Type.**

```python
sales.groupby('Item Type')['Order ID'].count()
```
Output:
```
Baby Food           7
Beverages           8
Cereal              7
Clothes            13
Cosmetics          13
Fruits             10
Household           9
Meat                2
Office Supplies    12
Personal Care      10
Snacks              3
Vegetables          6
```

**Q14. Which Item Type occurs most frequently?**

```python
sales['Item Type'].mode()
```
Output: `Clothes` and `Cosmetics`, tied at 13 orders each (matches the counts above). Genuinely a tie in this dataset, not a bug, `.mode()` correctly returns both when there's more than one most-frequent value.

**Q15. Orders per Sales Channel.**

```python
sales['Sales Channel'].value_counts()
```
Output: `Offline: 50`, `Online: 50` — an exact 50/50 split.

**Q16. Orders per Order Priority.**

```python
sales['Order Priority'].value_counts()
```
Output: `H: 30`, `L: 27`, `C: 22`, `M: 21`.

**Q17. Average value of every numerical column.**

```python
sales[nums_cols].mean()
```
Output:
```
Order ID         5.550204e+08
Units Sold       5.128710e+03
Unit Price       2.767613e+02
Unit Cost        1.910480e+02
Total Revenue    1.373488e+06
Total Cost       9.318057e+05
Total Profit     4.416820e+05
```

**Q18. Minimum and maximum of every numerical column.**

```python
sales[nums_cols].min()
```
I only ran `.min()` here and never followed up with `.max()`. That's a gap, for full revision, the missing half is just:

```python
sales[nums_cols].max()
```

**Q19. Total sales/revenue across the entire dataset.**

This one got mixed up with something else in my notebook. What I actually wrote was:

```python
sum = float(sales['Total Revenue'].sum())
number = int(sales['Order ID'].count())
print(sum/number)
```
Output: `1373487.6831`

This computes **average revenue per order** (sum divided by count), which is really just proving that `sales['Total Revenue'].mean()` works the way you'd expect (compare it to the mean from Q17, they're identical). It does **not** answer Q19. The actual total revenue across the dataset is simply:

```python
sales['Total Revenue'].sum()
```
which would return one big number (roughly the sum of all seven region totals from Q20 below). Worth adding this single line for a complete revision.

**Q20. Total revenue per Region.**

```python
sales.groupby('Region')['Total Revenue'].sum()
```
Output:
```
Asia                                 21347091.02
Australia and Oceania                14094265.13
Central America and the Caribbean     9170385.49
Europe                               33368932.11
Middle East and North Africa         14052706.58
North America                         5643356.55
Sub-Saharan Africa                   39672031.43
```

**Q21. Total revenue per Item Type.**

```python
items = sales.groupby('Item Type')['Total Revenue'].sum()
```
I did write this line, but never displayed it (no `print(items)` or bare `items` on its own line), so it never showed an output in the notebook. It got used later purely to build a bar chart. For revision, remember: **assigning a groupby result to a variable silently computes it but shows nothing until you actually display it.** Just add `items` on its own line, or `print(items)`, to see the numbers.

**Q22. Country with the highest total sales.**

```python
sales.groupby('Country')['Total Revenue'].sum().idxmax()
```
Output: `'Honduras'`.

**Q23. Country with the lowest total sales.**

This one was skipped entirely. The fix is a one-line mirror of Q22:

```python
sales.groupby('Country')['Total Revenue'].sum().idxmin()
```
`.idxmax()` and `.idxmin()` are a natural pair, worth always writing both together as a habit.

**Q24. Average sales for each Sales Channel.**

Also skipped. This is a groupby + mean, similar in shape to Q20:

```python
sales.groupby('Sales Channel')['Total Revenue'].mean()
```

**Q25. Total sales for each Region + Item Type combination.**

```python
sales.groupby(['Region', 'Item Type'])['Total Revenue'].sum().reset_index()
```
This is the classic multi-column `groupby()`, passing a *list* of columns instead of one groups by every combination of the two. `.reset_index()` turns the grouped result back into a flat, regular-looking DataFrame instead of a multi-level index, purely for readability.

---

## 🟠 Level 3 — NumPy Practice

**This entire section was skipped in my notebook.** I never converted a column to a raw NumPy array or used `np.` functions directly on the sales data, everything so far used Pandas' built-in methods instead (`.mean()`, `.min()`, etc.) which quietly use NumPy underneath anyway. Since the whole point of this level is deliberately practicing the NumPy layer directly, here's the code I should add, using `Total Revenue` as the example column:

```python
# Q26: convert a column to a NumPy array
revenue = sales['Total Revenue'].to_numpy()

# Q27: basic statistics
mean_rev = np.mean(revenue)
median_rev = np.median(revenue)
std_rev = np.std(revenue)
min_rev = np.min(revenue)
max_rev = np.max(revenue)

# Q28: percentiles
p25 = np.percentile(revenue, 25)
p50 = np.percentile(revenue, 50)
p75 = np.percentile(revenue, 75)

# Q29 & Q30: how many values are above / below the mean
above_mean = np.sum(revenue > mean_rev)
below_mean = np.sum(revenue < mean_rev)

# Q31: normalize (z-score) the column
normalized = (revenue - mean_rev) / std_rev

# Q32 & Q33: index of the max / min value
max_index = np.argmax(revenue)
min_index = np.argmin(revenue)
```

A couple of things worth remembering here, since they trip people up early on:

- `df['col'].to_numpy()` (or the older `.values`) is the standard way to drop out of Pandas into a plain NumPy array.
- `np.median()` exists as a separate function because there's no `np.mean`-style shortcut on arrays for the median, you always call it as `np.median(array)`, never `array.median()` like Pandas allows.
- `np.percentile(array, 50)` and `np.median(array)` always give the identical result, the median *is* the 50th percentile, just under a different name.
- `np.argmax()` / `np.argmin()` give you the **position** (index) of the extreme value, not the value itself. If you want the actual revenue amount, that's still just `max_rev` from `np.max()`.

---

## 🔵 Level 4 — Matplotlib Practice

**Q34. Histogram of a numerical column.**

```python
sales[nums_cols].hist(bins=10, figsize=(12, 8), edgecolor='white')
plt.tight_layout()
plt.show()
```
I plotted all numerical columns at once here rather than experimenting with one column across `bins=5`, `10`, `20` as suggested. Worth doing that comparison separately sometime, since it's a genuinely useful lesson: too few bins can hide real structure (like a second cluster of values), while too many bins can make the histogram look noisy and jagged even when the underlying pattern is smooth.

**Q35. Bar chart of orders per Region.**

```python
region_counts = sales.groupby('Region')['Order ID'].count()
region_counts.plot(kind='bar', x='Region', y='Order ID')
plt.title('Number of Orders by Region')
plt.ylabel('Order Count')
plt.show()
```

**Q36. Bar chart of total sales per Item Type.**

```python
items = sales.groupby('Item Type')['Total Revenue'].sum()
itemsidx = items.index
itemsvals = items.values

plt.figure(figsize=(10, 6))
plt.bar(itemsidx, itemsvals, color='skyblue', edgecolor='black')
plt.title('Total Sales by Item Type')
plt.xlabel('Item Type')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

**Q37. Pie chart of Sales Channel proportions.**

```python
channel = sales['Sales Channel'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(channel.values, labels=channel.index, autopct="%1.1f%%", startangle=90,
        colors=['#4CAF50', '#FF9800'], wedgeprops={'edgecolor': 'black'})
plt.show()
```
Since Offline and Online are exactly 50/50, this pie chart will show two perfectly even halves.

**Q38. Line graph of sales over time.**

```python
sales['Order Date'] = pd.to_datetime(sales['Order Date'])
daily_sales = sales.groupby('Order Date')['Order ID'].count()

plt.figure(figsize=(12, 5))
plt.plot(daily_sales.index, daily_sales.values, color='blue', linewidth=1.5, marker='o', markersize=3)
plt.title('Total Orders Over Time')
plt.xlabel('Order Date')
plt.ylabel('Number of Orders')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```
Worth a small note for revision: this tracks the **count of orders** over time, not revenue over time. If the question really means "sales" as in money, the same pattern works by swapping the aggregation: `sales.groupby('Order Date')['Total Revenue'].sum()`.

**Q39. Scatter plot between two numerical variables.**

```python
x_data = sales['Units Sold']
y_data = sales['Total Profit']

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='purple', alpha=0.5, s=30)
plt.title('Relationship Between Units Sold and Total Profit')
plt.xlabel('Units Sold')
plt.ylabel('Total Profit')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```
The question suggested Units Sold vs Total Revenue specifically, I used Total Profit instead, which is a perfectly reasonable substitution (it's still "two numerical variables"), just noting the swap here so I remember which pair I actually plotted.

---

## 🔴 Level 5 — EDA Questions

This is the level where actually writing down what you notice matters more than the code. Here's what the outputs above already tell me, without needing new code:

**Categorical analysis (Q44–46):**
- Sub-Saharan Africa has the most orders (36, from Q12) **and** the highest total revenue (39.67M, from Q20). In this dataset the two happen to agree, the region with the most orders is also the region bringing in the most money. That's worth noting specifically because it's not guaranteed to be true, a region could easily have lots of small orders but low total value, or few orders that are each very large.

**Product analysis (Q47–50):**
- Clothes and Cosmetics are tied for the most frequently ordered item types (13 orders each, from Q14).
- Which item type generates the *most revenue* is still an open question in my notebook, remember the Q21 gap above, the `items` variable was computed but never displayed. That's the very next thing to check to answer this properly, and to see whether the most-ordered item type is also the most profitable one.

**Sales channel analysis (Q51–53):**
- Orders are split exactly 50/50 between Offline and Online (from Q15), so "which channel has more orders" has a clean answer: neither, they're tied.
- Which channel earns more *average* revenue per order is still open too, that's the Q24 gap above.

**Distribution questions (Q40–43):** these were explored visually rather than with written conclusions:

```python
for col in nums_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(sales[col], kde=True, color='blue', bins=20)
    plt.title(f'Distribution Shape for {col}')
    plt.show()
```
This loop plots a histogram with a smoothed density curve for every numerical column in one go, a nice shortcut over writing seven separate `plt.hist()` calls. The follow-up I still need to actually write down: looking at each shape, note which columns look roughly bell-shaped (normal) versus which lean heavily to one side (skewed), and which have points sitting far away from the rest (possible outliers). The code produces the picture, but Q40 to Q43 are only really "answered" once I write a one-line conclusion under each plot.

---

## 🔥 Level 6 — Real EDA Challenge

**What was done:**
- ✅ 2+ distribution plots (the histogram loop above, and the combined `.hist()` grid from Q34)
- ✅ 3+ bar charts (Region orders, Item Type revenue, and the earlier Region orders chart again)
- ✅ 1 line plot (orders over time)
- ✅ 1 scatter plot (Units Sold vs Total Profit)
- ✅ 1 correlation analysis:

```python
correlation_matrix = sales[nums_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()
```

**Still missing: the 5 written observations.** All the charts exist, but the challenge specifically asks for conclusions in plain English, things like the Sub-Saharan Africa note above. That final step, translating a chart into a one-line insight, is really the entire point of EDA, and it's the one part that can't be copy-pasted, it has to come from actually looking at each plot.

---

## Summary: gaps to fix next time I revise this

1. **Level 3 (NumPy on raw arrays)** — skipped completely, code to add is in that section above.
2. **Q18** — only did `.min()`, still need `.max()`.
3. **Q19** — accidentally computed average revenue per order instead of total revenue; still need `sales['Total Revenue'].sum()`.
4. **Q21** — `items` variable computed but never displayed, just add `print(items)`.
5. **Q23** — never ran `.idxmin()` to find the lowest-revenue country.
6. **Q24** — never computed average revenue per Sales Channel.
7. **Q9** — used `.size` instead of true memory usage; correct version is `sales.memory_usage(deep=True).sum()`.
8. **Level 5/6 written observations** — the charts exist, the one-line conclusions under each one still need to be written out.

---

*Plan is to close each of these eight gaps next practice session, then this notes file is genuinely complete.*
