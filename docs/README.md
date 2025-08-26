# Definable Documentation

This directory contains the official documentation for the Definable backend platform, powered by [Mintlify](https://mintlify.com).

## Local Development

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Setup

1. Install the Mintlify CLI globally:

```bash
npm install -g @mintlify/cli
```

Or use the provided installation script:

```bash
./install.sh
```

2. Start the development server:

```bash
mintlify dev
```

This will start a local server at [http://localhost:3000](http://localhost:3000) with hot reloading enabled.

## Documentation Structure

- `docs.json` - Main configuration file
- `pages/` - Documentation content organized by sections
- `images/` - Diagrams and illustrations
- `logo/` - Brand assets

## Adding Content

All documentation pages are written in MDX, which combines Markdown with JSX components.

Create a new page by adding an MDX file to the appropriate directory in `pages/` and then update the navigation in `docs.json`.

### Page Structure

Each page should include frontmatter at the top:

```mdx
---
title: 'Page Title'
description: 'Page description for SEO and previews'
---

# Your Content Here
```

## Deployment

The documentation is automatically deployed when changes are pushed to the main branch. For manual deployment, contact the documentation team.

## Resources

- [Mintlify Documentation](https://mintlify.com/docs)
- [MDX Syntax](https://mdxjs.com/docs/what-is-mdx)
- [Definable Development Guidelines](https://github.com/definable.ai-io/guidelines) 