# Blogs (Markdown)

## How this folder works

- Each blog is a **Markdown file**.
- Each Markdown file must include **front matter** at the top.
- The site will automatically list blogs from the folder and render the correct Markdown file based on URL.

## Front matter schema

```yaml
---
title:
slug:
date:
category:
readingTime:
description:
tags:
  -
featuredImage:
author: Ujjwal Singh
---
```

## Add a new blog (under 1 minute)

1. Create `blogs/<your-slug>.md`
2. Copy the front matter block
3. Fill in metadata
4. Write the blog content in Markdown

## Rendering requirements

- Uses `marked.js` (GitHub-flavored Markdown)
- Uses `highlight.js` for code highlighting
- Table of Contents is generated from `h2`/`h3` headings
