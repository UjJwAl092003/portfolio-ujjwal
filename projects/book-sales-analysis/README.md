# Book Sales & Ratings Analysis

An exploratory data analysis of a multi-genre book catalog, digging into pricing, genre performance, author reputation, and sales trends to find insights a publisher could actually act on.

## Project Overview

Publishers sit on catalogs full of data, but rarely have time to ask it hard questions. This project takes a real book sales and ratings dataset and pushes past simple charts to answer things a publishing team actually cares about: which genres and languages the catalog depends on, whether an author's reputation tier actually predicts how well their books do, how price relates to volume, and which publishers are winning on revenue. Every chart in this project is paired with a business takeaway rather than left to speak for itself.

## Dataset

**Source:** Books Sales and Ratings dataset (Kaggle) — `Books_Data_Clean.csv`
**Size:** ~988 books after cleaning

Each row is a single book, with fields covering publishing year, author, genre, language, sale price, units sold, gross sales, publisher revenue, and reader ratings (average rating and ratings count), plus an author reputation tier (Novice, Intermediate, Famous, Excellent).

Cleaning involved removing records with unrealistic publishing years (including a handful representing ancient texts with negative years), dropping rows missing a book title, and standardizing column names. No duplicate records were found.

## Tools Used

- **Python** — core language
- **Pandas** — data cleaning and aggregation
- **NumPy** — numerical operations
- **Matplotlib** — core plotting
- **Seaborn** — statistical visualizations
- **Jupyter Notebook** — analysis environment

## Key Insights

- **Author reputation tier is a weak signal.** Authors labeled "Excellent" do *not* get the most reader engagement — "Intermediate" and "Famous" tier authors actually average more ratings, a genuinely counterintuitive result.
- **Genre concentration risk.** Roughly three-quarters of the catalog is a single genre, "genre fiction," leaving little diversity if demand in that category softens.
- **The catalog is almost entirely English-language**, with translated editions representing a largely untapped growth opportunity.
- **Rating quality doesn't guarantee reach.** Average rating and ratings count are only loosely related — a highly rated book isn't guaranteed a large audience.
- **Unit sales are price-sensitive**, clustering heavily at lower price points and tapering off sharply as price rises.
- **Digital distribution is already a top-3 competitor.** Amazon Digital Services ranks third in total publisher revenue, ahead of HarperCollins and Hachette.
- **Revenue follows a long tail.** A small handful of authors account for a disproportionate share of total gross sales.

## Key Visualizations

### Genre Distribution

The catalog is dominated by a single genre, exposing a real concentration risk if reader demand shifts.

### Author Reputation vs. Reader Engagement

Units sold does not scale cleanly with an author's labeled reputation tier — the core counterintuitive finding of this project.

### Price vs. Units Sold

Sales volume concentrates sharply at lower price points, a clear signal for pricing strategy on volume-driven titles.

### Top Authors by Revenue

A small number of authors generate a disproportionate share of total gross sales, a classic long-tail pattern.

---

**Full write-up, complete notebook, and business recommendations:** see the project repository.
