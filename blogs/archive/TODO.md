# TODO - Blog categories system

## Step 1: Update manifest categories

- Update `blogs/manifest.json` category values for each existing blog to the new category model.
- Preserve dates.

## Step 2: Update blogs.html category dropdown + filtering

- Replace hardcoded/implicit category system with dynamic category population from `blogs/manifest.json`.
- Ensure selecting a category filters blogs by `blog.category`.

## Step 3: Verify

- Confirm category dropdown shows the new categories.
- Confirm filtering shows correct blogs.
- Confirm blog ordering remains newest-first.
- Confirm blog detail rendering + syntax highlighting still works.

## Step 4: Verify “no changes” constraints

- Ensure no publication dates are modified further.
- Ensure no blog entries are removed.
