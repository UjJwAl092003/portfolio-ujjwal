# Confidence Intervals — A Beginner-Friendly Guide

> These notes explain Confidence Intervals (CI) from absolute scratch. No prior statistics background needed — just read top to bottom in order.

---

## Table of Contents
1. [Warm-up: Population vs Sample](#1-warm-up-population-vs-sample)
2. [Parameter vs Statistic](#2-parameter-vs-statistic)
3. [Inferential Statistics — the big picture](#3-inferential-statistics--the-big-picture)
4. [Point Estimate](#4-point-estimate)
5. [What is a Confidence Interval? (formal definition)](#5-what-is-a-confidence-interval-formal-definition)
6. [Two ways to calculate a CI](#6-two-ways-to-calculate-a-ci)
7. [Method 1: Z-Procedure](#7-method-1-z-procedure)
8. [Rigorous derivation of the Z-formula](#8-rigorous-derivation-of-the-z-formula)
9. [Fully worked Z-procedure example (2D:4D ratio study)](#9-fully-worked-z-procedure-example-2d4d-ratio-study)
10. [How to correctly interpret a Confidence Interval](#10-how-to-correctly-interpret-a-confidence-interval)
11. [Two classic misinterpretations to avoid](#11-two-classic-misinterpretations-to-avoid)
12. [What affects the width of a CI?](#12-what-affects-the-width-of-a-ci)
13. [Method 2: T-Procedure](#13-method-2-t-procedure)
14. [The t-distribution explained in depth](#14-the-t-distribution-explained-in-depth)
15. [Fully worked T-procedure example (cereal box weights)](#15-fully-worked-t-procedure-example-cereal-box-weights)
16. [Bonus: Confidence Interval for a Proportion](#16-bonus-confidence-interval-for-a-proportion)
17. [Worked example in Python](#17-worked-example-in-python)
18. [Quick-reference cheat sheet](#18-quick-reference-cheat-sheet)

---

## 1. Warm-up: Population vs Sample

Imagine you want to know the **average height of every male student in a university**.

- **Population** → *every single* student in that university (the entire group you care about).
- **Sample** → a smaller, manageable group of students picked *randomly* from that population.

Why do we bother with a sample at all? Because measuring the *entire* population is usually:
- too expensive,
- too slow, or
- literally impossible (e.g., you can't interview every YouTube subscriber in the world, or every adult American).

So instead, we pick a random subset (say, 100 people), measure them, and use that to make a smart guess about the whole population.

**Two rules for a good sample:**
1. **Random** — every member of the population should have a fair chance of being picked. This is sometimes called a **Simple Random Sample (SRS)**. If the sampling design is biased, the end results can be very misleading, no matter how good the math afterward is.
2. **Representative** — it should reflect the diversity of the population. (E.g., if you want India's average salary, don't sample people from just one state — sample across all states.)

```
┌─────────────────────────────┐
│         POPULATION          │   <- everyone you care about
│   ┌───────────────────┐     │
│   │      SAMPLE        │    │   <- small random, representative subset
│   │  (what you actually │    │
│   │   collect data on)  │    │
│   └───────────────────┘     │
└─────────────────────────────┘
```

---

## 2. Parameter vs Statistic

Once you understand population vs sample, this next distinction becomes easy — it's just about *naming*.

| | Belongs to | Called | Usually written using |
|---|---|---|---|
| **Parameter** | Population | Parameter | Greek letters: mean = μ (mu), std dev = σ (sigma), proportion = p |
| **Statistic** | Sample | Statistic | Roman letters: mean = x̄ (x-bar), std dev = s, proportion = p̂ (p-hat) |

**Key real-world fact:** Parameters (population values) are almost always **unknown** — it's usually impossible to measure an entire population. So we *estimate* the parameter using a statistic calculated from a sample.

Example: You want to know μ (population mean height), but you can't measure everyone, so you calculate x̄ (sample mean height) instead and use it as your best guess for μ.

---

## 3. Inferential Statistics — the big picture

Statistics has two broad branches:

- **Descriptive statistics** — just describing/summarizing the data you already have.
- **Inferential statistics** — using a *sample* to draw conclusions (make inferences/predictions) about the *population*.

Confidence Intervals belong to **inferential statistics**. Other techniques in this same family (which you'll meet later) include:
- Hypothesis testing
- Regression analysis (like linear regression in ML)

Typical questions inferential statistics helps answer:
- Is there a real difference between two groups?
- Can I predict variable Y using variable X?
- Given my sample mean, what can I say about the population mean?

---

## 4. Point Estimate

A **point estimate** (or **point estimator**, before we actually draw the sample) is simply: **a single value, calculated from your sample, that serves as your best guess for an unknown population parameter.**

The sample mean x̄ is called a **point estimator** of the population mean μ because once we actually collect our sample, x̄ takes on one single value (e.g. 37.3), and that single value is called a **point estimate** of μ.

### Example
Say you run a YouTube channel with **77,000 subscribers** (this is your population), and you want to know the average age of all subscribers.

You can't ask all 77,000 people. So instead:
1. You run one live class with 100 random students.
2. You ask everyone to type their age in the chat.
3. You calculate the average → say it comes out to **28**.

That number, **28**, is your **point estimate** of the true (unknown) population mean age.

> Point estimates aren't limited to the mean — you can also compute a point estimate for standard deviation, variance, or proportion. But mean is the most common one discussed.

### The problem with a point estimate alone
In statistics, a lone point estimate isn't considered very useful **without some measure of the uncertainty attached to it**. How close is the sample mean likely to be to the real population mean? A single number gives us no sense of that — which is exactly the gap a Confidence Interval fills.

### Making the point estimate stronger
You could repeat the sampling process multiple times (e.g., run 10 different live classes, get 10 different sample means x̄₁, x̄₂, ... x̄₁₀), then average those 10 numbers together. This gives a more stable estimate — but it's still just a **point estimate** (a single number). This trick is essentially applying the **Central Limit Theorem** (more on that soon).

---

## 5. What is a Confidence Interval? (formal definition)

Confidence intervals are based on sample data, and they give **a range of plausible values for a parameter** — instead of relying on one single (fragile) number.

> **Confidence Interval (CI):** A range of values, calculated from sample data, within which we believe the true (unknown) population parameter lies, stated together with a confidence level that expresses how sure we are.

### Formal examples
- We may be **95% confident** that the population mean μ lies in the interval **−0.2 to 3.1**. Here, −0.2 to 3.1 is a 95% confidence interval for μ.
- We may be **99% confident** that the population standard deviation σ lies in the interval **2.5 to 13.4**. Here, 2.5 to 13.4 is a 99% confidence interval for σ.

### The single most important rule of confidence intervals

> **A confidence interval is always constructed for a parameter — never for a statistic.**

We will *never* say "here is a confidence interval for x̄." That is simply not how it works. We may *use* a statistic (like x̄ or p̂) to *help us calculate* the interval, but the interval itself is always a statement about the unknown population parameter (μ, σ, or p).

### Confidence Level
Associated with every confidence interval is a **confidence level** (e.g. 95%, 99%). This tells us how confident we are that the interval actually contains the true parameter.

- We can pick the confidence level to be *anything* we like — 90%, 95%, 99%, or even something unusual like 23.7%.
- **There is a trade-off:** a higher confidence level always produces a **wider** interval. A 99% CI will be wider than a 95% CI, for the same data.
- **95% is by far the most common choice** in practice — it's considered a reasonable balance between being confident and having a usably narrow range.

### General form of many confidence intervals
Many statistics (like the sample mean) have sampling distributions that are approximately Normal — largely because of the **Central Limit Theorem**. In these cases, the CI takes the general form:

```
Confidence Interval = Point Estimate  ±  Margin of Error
```

Not every confidence interval is built this way, but a large number of the common ones are.

### Two big tasks whenever we build a CI
1. Use an **appropriate method** to construct the interval (this depends on what you know/don't know — covered in the Z- and T-procedures below).
2. **Properly interpret** the resulting interval (this has some subtle traps — covered in Section 10 & 11).

---

## 6. Two ways to calculate a CI

There are two common procedures used to calculate a Confidence Interval for a population mean:

| Procedure | When to use it |
|---|---|
| **Z-procedure** | When you *know* the population standard deviation (σ) — rare in practice |
| **T-procedure** | When you *don't* know σ (this is the realistic, everyday case) |

We learn the Z-procedure first because it builds the theoretical foundation — even though in real life, you'll almost always end up using the T-procedure.

> **Important rule to always remember:** A Confidence Interval is always about the **population parameter**, never about the sample statistic. The sample just *helps* you estimate it.

CIs show up everywhere — business, finance, economics, and the media. A very common real-world example is a **presidential approval rating poll**: pollsters never claim to know the *exact* percentage of the population who approve — they always report a range with a margin of error (see Section 16). You may have already seen CIs without realizing it — e.g., the "error bars" on bar charts in libraries like Seaborn are actually confidence intervals!

---

## 7. Method 1: Z-Procedure

### Assumptions (all must hold to use this method)
1. The sample was drawn **randomly** (a simple random sample) from the population of interest — good sampling design matters a lot; a biased design can make the whole analysis misleading.
2. You know the **population standard deviation (σ)**. *(This is going to be rare in practice — usually we must estimate σ using sample data, which is exactly why the T-procedure exists.)*
3. The **population distribution is Normal**.
   - *Relaxation:* if the population isn't normal but your **sample size is large**, the **Central Limit Theorem (CLT)** takes care of things — the sampling distribution of the mean will still be approximately normal.

### The formula

```
CI = x̄  ±  Z(α/2) × ( σ / √n )
```

Where:
- **x̄** = sample mean (your point estimate)
- **σ** = population standard deviation (known)
- **n** = sample size
- **Z(α/2)** = the **critical value** — the z-value that has an area of α/2 to its right under the standard normal curve
- **σ/√n** = the true standard deviation of the sampling distribution of x̄ (sometimes written **σₓ̄**, and called the **standard error** when estimated from a sample)

### Visualizing where Z(α/2) comes from

For a `(1 − α)×100%` confidence interval, we place an area of `(1 − α)` in the middle of the standard normal curve, and split the remaining area `α` evenly into the two tails (`α/2` on each side).

![Standard Normal Distribution with 95% critical values](images/z_distribution_critical_values.png)

### Common Z-values for popular confidence levels

| Confidence Level | α | α/2 | Z(α/2) value |
|---|---|---|---|
| 90% | 0.10 | 0.05 | 1.645 |
| 95% | 0.05 | 0.025 | **1.96** |
| 99% | 0.01 | 0.005 | 2.576 |

Notice the trade-off directly in this table: if we're willing to accept a **lower** confidence level, what we get back in return is a **smaller** margin of error and a **narrower** interval — and vice versa.

### Quick Example
- Point estimate (x̄) = 28
- Population standard deviation (σ) = 50 (assumed known here)
- Sample size (n) = 100
- Desired confidence level = 95% → Z(α/2) = **1.96**

```
Margin of Error = 1.96 × (50 / √100) = 1.96 × 5 = 9.8

CI = 28 ± 9.8  →  [18.2, 37.8]
```

So we'd say: *"I am 95% confident the true population mean lies between 18.2 and 37.8."*

---

## 8. Rigorous derivation of the Z-formula

This section shows precisely *why* the formula looks the way it does, step by step.

### Step 1 — Set up the sampling scenario
Suppose we are about to draw a random sample of **n independent observations** from a normally distributed population with mean **μ** and standard deviation **σ**.

Even *before* we draw the sample, we can treat the sample mean **x̄** as a random variable. Under these conditions, x̄ is itself normally distributed with:
- mean = μ
- standard deviation = **σ/√n** (this quantity is sometimes written **σₓ̄**, and is the true standard deviation of the sampling distribution of x̄)

### Step 2 — Standardize x̄
Just like any normal random variable can be converted to the standard normal, we standardize x̄:

```
Z  =  (x̄ − μ) / (σ/√n)
```

This new random variable **Z** has the **standard normal distribution** (mean 0, std dev 1).

### Step 3 — Recall a key property of the standard normal distribution
If Z is a standard normal random variable, then:

```
P( −1.96 < Z < 1.96 ) = 0.95
```

Generalizing this for *any* confidence level:

```
P( −Z(α/2) < Z < Z(α/2) )  =  1 − α
```

where **Z(α/2)** is defined as *the z-value that has an area of α/2 to its right* under the standard normal curve.

For a 95% CI: α = 0.05, so α/2 = 0.025, and Z(0.025) = 1.96 (because 1.96 has exactly an area of 0.025 to its right).

### Step 4 — Substitute our standardized x̄ into this probability statement

Since `Z = (x̄ − μ)/(σ/√n)`, we substitute it directly:

```
P( −Z(α/2)  <  (x̄ − μ)/(σ/√n)  <  Z(α/2) )  =  1 − α
```

### Step 5 — Use algebra to isolate μ

We want a statement about **μ**, not about x̄. Multiplying through by `(σ/√n)` and rearranging (flipping the inequality direction where needed) gives:

```
P( x̄ − Z(α/2)×(σ/√n)   <   μ   <   x̄ + Z(α/2)×(σ/√n) )  =  1 − α
```

### The crucial subtlety
Here, **x̄ is the random variable** — it's the thing that varies from sample to sample. **μ is a fixed, unknown constant** that we are trying to estimate; it does not vary. All the "randomness" in this probability statement comes from x̄, not from μ.

We call:
- `x̄ − Z(α/2)×(σ/√n)` the **lower bound** of the confidence interval
- `x̄ + Z(α/2)×(σ/√n)` the **upper bound** of the confidence interval

### Final result

```
A (1−α)×100% confidence interval for μ is:

     x̄  ±  Z(α/2) × (σ/√n)
```

For a 95% CI specifically: α = 0.05, Z(0.025) = 1.96, so:

```
95% CI  =  x̄  ±  1.96 × (σ/√n)
```

That's exactly our formula — derived cleanly from the standard normal distribution and the sampling distribution of x̄.

---

## 9. Fully worked Z-procedure example (2D:4D ratio study)

This is a real style of study used to show every step end-to-end, including checking assumptions.

**Background:** The 2D:4D ratio is the ratio of the length of the index finger (2nd digit) to the ring finger (4th digit). Researchers sometimes study whether this ratio relates to traits like physical aggression.

**Data:** A sample of **135 women** of European descent at a large college yielded a sample mean ratio of **x̄ = 0.988**. Suppose it is known that the population standard deviation is **σ = 0.028** (unusual to know this in real life, but assumed here for the sake of the example). We want a **95% confidence interval** for the population mean μ.

### Step 1 — Check the assumptions
1. **Random sample?** Assume yes.
2. **Normal population, or large enough sample?** A boxplot and a Normal Quantile-Quantile (Q-Q) plot of the data are checked. A Q-Q plot that comes out as an approximately straight line indicates the data is approximately normally distributed. Here it looks reasonably straight, so the normality assumption is considered acceptable. (And in any case, with n = 135, which is well over 30, the Central Limit Theorem would rescue us even if the population were mildly non-normal.)

### Step 2 — Plug into the formula
```
CI = x̄  ±  Z(0.025) × (σ/√n)
   = 0.988  ±  1.96 × (0.028/√135)
   = 0.988  ±  0.0047
```

### Step 3 — Compute the bounds
```
Lower bound = 0.988 − 0.0047 ≈ 0.983
Upper bound = 0.988 + 0.0047 ≈ 0.993
```

**95% CI for μ = [0.983, 0.993]**

### Step 4 — Interpret it in context
*"We can be 95% confident that the true mean 2D:4D ratio for women of European descent at this college lies between 0.983 and 0.993."*

**Important caveat:** Strictly speaking, this conclusion only formally applies to the *population from which the sample was drawn* (women of European descent at *this particular college*). It may hint at what's true for women of European descent more broadly, but we should always ask: *where did the sample come from, and to what population do the conclusions really apply?*

---

## 10. How to correctly interpret a Confidence Interval

This is where **most people get it wrong** — and it's a favorite interview question.

The population mean μ is a **parameter**. It has a **fixed value that is usually unknown** — it is not random, it does not change.

### The short (but slightly unsatisfying) interpretation
For a 95% CI of [4.8, 22.7]:
> "We can be 95% confident that the parameter μ lies within the interval [4.8, 22.7]."

This is correct, but what precisely do we *mean* by "95% confident"? Let's dig deeper.

### The repeated-sampling interpretation (the real meaning)
Imagine repeating the entire process — draw a random sample, calculate a 95% CI — over and over again, many times.

> **95% of these repeatedly-calculated 95% confidence intervals would capture (contain) the true value of μ. The other 5% would miss it.**

Some intervals will land squarely around μ; occasionally, due to an unusually small or large sample mean, an interval will completely miss μ.

![Repeated sampling confidence intervals](images/ci_repeated_sampling.png)

In the picture above, the black vertical line represents the true (fixed, usually unknown) population mean μ. Each horizontal line is one 95% CI from one random sample — the dot in the middle is that sample's x̄. **Blue** intervals captured μ; **red** intervals missed it. Roughly 95% of many such intervals will be blue.

### In practice
In real life, we only draw **one** sample, so we only get **one** confidence interval. We never know for certain whether *our particular* interval is one of the "lucky" ones that captured μ, or one of the "unlucky" ones that missed. But we can say: **we are 95% confident that our interval is one of the ones that captured μ.**

### Worked example: Body Mass Index (BMI)
A random sample of **28 female students** at a university had a sample mean BMI of **24.7**, and the corresponding 95% confidence interval for μ was found to be **[22.6, 26.8]**.

**Proper interpretation:**
> "We can be 95% confident that the mean BMI of *all* female students at this university lies between 22.6 and 26.8."

Equivalently:
> "In repeated sampling, 95% of the 95% confidence intervals calculated this way would capture the true mean BMI of female students at this university. Since we have one of those intervals, we can be 95% confident that our interval captures μ."

---

## 11. Two classic misinterpretations to avoid

Using the BMI example above (95% CI = [22.6, 26.8] for mean BMI of female students at a university), here are two very common **wrong** interpretations:

### ❌ Misinterpretation 1
> "95% of female students at this university have a BMI between 22.6 and 26.8."

**Why it's wrong:** The confidence interval says nothing about the spread of *individual* BMI values across students. It is a statement purely about the **population mean** — a single summary number — not about what percentage of individuals fall in that range.

### ❌ Misinterpretation 2
> "There is a 95% probability that the sample mean BMI lies between 22.6 and 26.8."

**Why it's wrong:** This confuses the sample mean (a statistic, x̄, which we've already observed to be exactly 24.7 — no uncertainty left about it) with the population mean (the parameter, μ, which is unknown). The confidence interval is a statement about **μ**, never about **x̄**.

> **The golden rule, once again: a proper interpretation of a confidence interval always relates to the value of a *parameter*, and never to the value of a *statistic*.**

---

## 12. What affects the width of a CI?

Recall the formula:
```
CI = x̄  ±  Z(α/2) × (σ / √n)
```

Three things control the **width** (margin of error) of the interval:

### (a) Confidence Level (affects the critical value)
- Higher confidence level → **larger** Z(α/2) or t(α/2) → **wider** interval.
- If you wanted 100% confidence, your interval would need to stretch from −∞ to +∞ (completely useless, but "always right").
- If you wanted 0% confidence, your interval collapses to a single point (very precise, but almost certainly wrong).
- **95% is the industry-standard sweet spot** — reasonably narrow, reasonably confident. But 90%, 99%, or other levels are used depending on the context.

### (b) Population (or Sample) Standard Deviation
- Higher σ (or s) — more variability/spread in the data — → **wider** interval.
- Makes sense: if the underlying data is very spread out (inconsistent), your estimate naturally has to cover a wider range to be confident.
- Example: a very consistent batsman (small σ in scores) needs a narrow CI to predict his runs; an unpredictable batsman (large σ) needs a much wider CI.

### (c) Sample Size (n)
- Larger n → **narrower** interval (more data = more precision), because the margin of error shrinks proportional to `1/√n`.
- But there are **diminishing returns**: going from n=10 to n=30 shrinks the interval a lot; going from n=200 to n=500 barely shrinks it further.

![Margin of Error vs Sample Size](images/margin_of_error_vs_sample_size.png)

---

## 13. Method 2: T-Procedure

### The core problem with the Z-procedure
The Z-procedure assumed you *know* the population standard deviation (σ). In real life, this is almost never true — if you don't know the population mean, why would you know its standard deviation?

**Solution:** Use the **sample standard deviation (s)** instead of σ, and use the quantity `s/√n`, called the **standard error of x̄**, as our estimated standard deviation of the sampling distribution of x̄.

But this substitution introduces a new problem — the sample standard deviation itself varies from sample to sample, adding **extra uncertainty**. Once you replace σ with `s`, the quantity `(x̄ − μ)/(s/√n)` is **no longer standard normal** — it now follows the **Student's t-distribution**.

### Assumptions for T-procedure
1. Sampling must be a **simple random sample** from the population of interest. Good sampling design is always important.
2. The population should be **normally distributed** — but this assumption is not critical if the sample size is large (Central Limit Theorem again).
3. We do **not** need to know σ — that's the whole point of this procedure. We estimate it with the sample standard deviation, s.

### The T-formula

```
CI = x̄  ±  t(α/2, df) × ( s / √n )
```

Where:
- **s** = sample standard deviation (replaces σ)
- **t(α/2, df)** = critical value from the **t-distribution table** (instead of the Z-table) — the t-value with an area of α/2 to its right
- **df (degrees of freedom)** = **n − 1**

Notice this formula is structurally identical to the Z-formula — we've simply replaced σ with s, and Z with t.

---

## 14. The t-distribution explained in depth

### Where does it come from, mathematically?
When sampling from a normally distributed population, if we standardize using the *known* σ, we get:
```
Z = (x̄ − μ) / (σ/√n)      →  standard normal distribution
```
But if σ is unknown and we substitute the sample standard deviation s instead:
```
T = (x̄ − μ) / (s/√n)      →  t-distribution with (n − 1) degrees of freedom
```
The only thing that changed is swapping a **known constant** (σ) for an **estimated statistic** (s) — but that swap is enough to change the entire shape of the distribution, because s itself has its own sampling variability (it's different for every sample you draw).

### Who invented it, and why?
It was developed by a statistician who published under the pen name **"Student"** (hence "Student's t-distribution"), while working at a brewery — specifically to handle situations with **small sample sizes** and an **unknown population standard deviation**.

### What does it look like?
It looks similar to the normal (bell) curve — symmetric about zero, bell-shaped — but with **heavier tails** (more area in the tails) and a **lower peak**, to account for the added uncertainty of estimating σ using a sample statistic.

![t-distribution vs Normal distribution](images/t_vs_normal_distribution.png)

### Key parameter: Degrees of Freedom (df)
The exact shape of the t-distribution depends on just **one parameter**: `df = n − 1`. (This is directly tied to the fact that the sample variance formula also divides by `n − 1`.)

- **Small df (small sample size)** → fatter tails, lower peak → t-critical value is noticeably **larger** than the Z-critical value → **wider CI** (to compensate for the extra uncertainty of not knowing σ).
- **As df increases** → the t-distribution gets closer and closer to the standard normal distribution.
- **As df → ∞** → the t-distribution becomes *exactly* the standard normal distribution.

### Table: t-critical values for a 95% CI at various degrees of freedom

| Sample size (n) | Degrees of freedom (df = n−1) | t(0.025, df) |
|---|---|---|
| 2 | 1 | 12.706 |
| 6 | 5 | 2.571 |
| 11 | 10 | 2.228 |
| 21 | 20 | 2.086 |
| 31 | 30 | 2.042 |
| 101 | 100 | 1.984 |
| ∞ | ∞ | 1.960 (this is just Z(0.025)) |

Notice how, even at **df = 30**, the t-value (2.042) is still noticeably bigger than 1.96, and even at **df = 100** it's still slightly different from 1.96.

### An important myth to bust
Some sources claim: *"If your sample size is greater than 30, just forget about the t-distribution and use the standard normal distribution instead."* **This is not a good habit.** If you are using a standard deviation that is **based on sample data** (i.e., you don't actually know σ), you should keep using the t-distribution regardless of sample size — using Z instead of t when σ is estimated will make your calculated margin of error **smaller than it should be**, making your interval overconfident (falsely narrow).

---

## 15. Fully worked T-procedure example (cereal box weights)

**Background:** A cereal producer makes boxes with a stated weight of 750 grams. From a large lot of these boxes, a **random sample of 7 boxes** yielded:
- Sample mean, x̄ = **795.3 grams**
- Sample standard deviation, s = **17.8 grams**

We want a **95% confidence interval** for μ, the true mean weight of boxes in this lot.

### Step 1 — Decide which procedure to use
We do **not** know the population standard deviation σ here — only the sample standard deviation s. So we must use the **T-procedure**, not the Z-procedure.

### Step 2 — Check assumptions
The t-procedure assumes a normally distributed population. We should always check this before proceeding — e.g., with a Normal Q-Q plot. With only 7 data points, the plot shows an approximately straight line, so there's no glaring deviation from normality — but it's always a little uncertain to use these procedures with such a small sample size, since performance then depends heavily on the normality assumption actually holding.

### Step 3 — Find the critical t-value
- Sample size n = 7 → degrees of freedom df = n − 1 = **6**
- Confidence level = 95% → α = 0.05 → α/2 = 0.025
- Looking this up in a t-table (row df=6, column 0.025): **t(0.025, 6) = 2.447**

### Step 4 — Plug into the formula
```
CI = x̄  ±  t(0.025, 6) × (s/√n)
   = 795.3  ±  2.447 × (17.8/√7)
   = 795.3  ±  16.46
```

### Step 5 — Compute the bounds
```
Lower bound = 795.3 − 16.46 ≈ 778.8
Upper bound = 795.3 + 16.46 ≈ 811.8
```

**95% CI for μ = [778.8, 811.8] grams**

### Step 6 — Interpret in context
*"We can be 95% confident that the true mean weight of cereal in boxes from this lot lies between 778.8 and 811.8 grams."*

**Caveats:**
- This gives a hint about cereal boxes of this type in general, but statistically the conclusion applies only to **this specific lot** — the population we actually sampled from.
- It's genuinely a bit risky to use the t-procedure with such a tiny sample size (n = 7) — with small samples, results are heavily reliant on the normality assumption actually being true. A larger sample size would make us feel much more comfortable.

---

## 16. Bonus: Confidence Interval for a Proportion

CIs aren't just for means — they're extremely common for **proportions** too, especially in polling and surveys.

### Example: Presidential approval rating
Suppose, in a telephone poll of **1,000 adult Americans**, **430** said they approve of how the president is handling his job.

- These 1,000 people are our **sample**.
- **p̂ (p-hat)**, the **sample proportion**, is `430/1000 = 0.43`.
- **p**, the **population proportion** (the true proportion of *all* adult Americans who would say they approve if asked this way), is an unknown **parameter** we're trying to estimate.

The key question: how close is p̂ likely to be to the true p? Since p is unknown, it's conceivable — before we think it through — that p could be as different as 0.92 or 0.04. A confidence interval is exactly how we express our uncertainty about this.

### Reading a typical media statement
You'll often see something like:
> "The poll is believed to be accurate to within 0.03, **19 times out of 20**."

Breaking this down:
- "19 times out of 20" = 19/20 = **0.95** → a **95% confidence level**.
- "Accurate to within 0.03" → this 0.03 is the **margin of error**.

So this statement is really saying: *"We are 95% confident that p̂ will lie within 0.03 of the true value of p."*

### Building the interval
```
CI for p = p̂  ±  margin of error
         = 0.43  ±  0.03
         = [0.40, 0.46]
```

**Interpretation:** *"We can be 95% confident that the true population proportion p lies between 0.40 and 0.46."*

(The underlying formula for a proportion's margin of error involves `Z(α/2) × √(p̂(1−p̂)/n)` — very similar in spirit to the mean's formula, just with a different standard error expression. As with the mean, this interval is a statement about the fixed but unknown parameter **p**, never about the sample statistic **p̂**.)

---

## 17. Worked example in Python

Below is a simulation showing both procedures, plus a demonstration of the "95% of intervals contain the true mean" idea from Section 10.

```python
import numpy as np
import scipy.stats as stats

# ---------------------------------------------------------
# Example: Estimating average "fare" paid by Titanic passengers
# (using sample data to estimate an unknown population mean)
# ---------------------------------------------------------

np.random.seed(42)

# Suppose this is our full "population" of fares (in real life, unknown)
population_fares = np.random.exponential(scale=32, size=1300)  # skewed, non-normal population
true_population_mean = population_fares.mean()

print("True population mean (normally unknown!):", round(true_population_mean, 2))

# --- Step 1: Take repeated random samples & compute sample means (CLT in action) ---
sample_size = 30
num_samples = 10

sample_means = []
sample_stds = []

for i in range(num_samples):
    sample = np.random.choice(population_fares, size=sample_size, replace=False)
    sample_means.append(sample.mean())
    sample_stds.append(sample.std(ddof=1))  # ddof=1 for sample std dev

point_estimate = np.mean(sample_means)     # our best guess for population mean
sample_std_estimate = np.mean(sample_stds) # our best guess, since population std dev is unknown

print("Point estimate (x̄):", round(point_estimate, 2))

# --- Step 2: T-procedure (used because population std dev is unknown) ---
confidence_level = 0.95
alpha = 1 - confidence_level
df = sample_size - 1

t_critical = stats.t.ppf(1 - alpha/2, df)
margin_of_error = t_critical * (sample_std_estimate / np.sqrt(sample_size))

lower_bound = point_estimate - margin_of_error
upper_bound = point_estimate + margin_of_error

print(f"t-critical value: {round(t_critical, 3)}")
print(f"{int(confidence_level*100)}% Confidence Interval: [{round(lower_bound,2)}, {round(upper_bound,2)}]")

# ---------------------------------------------------------
# Bonus: Verifying the "95% of intervals contain true mean" idea
# ---------------------------------------------------------
hits = 0
trials = 200

for _ in range(trials):
    sample = np.random.choice(population_fares, size=sample_size, replace=False)
    mean = sample.mean()
    std = sample.std(ddof=1)
    margin = stats.t.ppf(1 - alpha/2, df) * (std / np.sqrt(sample_size))
    lo, hi = mean - margin, mean + margin
    if lo <= true_population_mean <= hi:
        hits += 1

print(f"Out of {trials} trials, {hits} intervals contained the true mean "
      f"({round(100*hits/trials,1)}%) — should be close to 95%")

# ---------------------------------------------------------
# Bonus: Confidence interval for a proportion (polling example)
# ---------------------------------------------------------
n_poll = 1000
approve = 430
p_hat = approve / n_poll

z_critical = stats.norm.ppf(0.975)  # for 95% CI
se_proportion = np.sqrt(p_hat * (1 - p_hat) / n_poll)
moe_proportion = z_critical * se_proportion

print(f"\nProportion CI: {round(p_hat,3)} ± {round(moe_proportion,3)} "
      f"→ [{round(p_hat-moe_proportion,3)}, {round(p_hat+moe_proportion,3)}]")
```

**Common beginner mistake to avoid:** Don't calculate the standard deviation of the *entire sampling distribution* (i.e., the std dev of your list of sample means) and plug that directly into the formula. Instead, use the **sample standard deviation of a single sample** (or average the sample std devs across your samples), because the formula already accounts for the `σ/√n` (or `s/√n`) adjustment. Mixing this up will give you a much narrower (and wrong) interval.

---

## 18. Quick-reference cheat sheet

| | Z-Procedure | T-Procedure |
|---|---|---|
| **Use when** | Population σ is known (rare) | Population σ is unknown (realistic case) |
| **Formula** | x̄ ± Z(α/2) × (σ/√n) | x̄ ± t(α/2, df) × (s/√n) |
| **Table used** | Z-table | t-table (needs df = n−1) |
| **Distribution shape** | Standard Normal | Student's t (fatter tails, especially for small n) |
| **Sample size needed** | Works well if n > 30 or population normal | Works even for small n; **use it regardless of n** if σ is estimated from data |

### The 3-step recipe (works for both procedures)
1. Calculate your **point estimate** (usually the sample mean, x̄).
2. Calculate the **margin of error**:
   - Z-procedure: `Z(α/2) × (σ/√n)`
   - T-procedure: `t(α/2, df) × (s/√n)`
3. Report: `CI = point estimate ± margin of error`, along with your chosen **confidence level**, and always give a **contextual interpretation** in terms of the population parameter.

### Remember
- A CI is about the **population parameter**, never the sample statistic.
- A 95% CI does **not** mean "95% probability the true mean is in this specific range." It means "if I repeat this process many times, 95% of the resulting intervals will contain the true mean."
- Higher confidence level → wider interval. Higher σ/s → wider interval. Larger sample size → narrower interval (with diminishing returns).
- If σ is estimated from sample data, use the **t-distribution** — not the normal distribution — no matter how large your sample is.
- Conclusions from a CI strictly apply only to the **population the sample was actually drawn from**.

---

*End of notes. Try re-running the Python example with different sample sizes and confidence levels to build intuition — seeing the numbers change in front of you is the fastest way to make this concept stick.*
