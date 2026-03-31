# Contributing Guide

The `rs-embed` project welcomes bug reports, documentation fixes, model integrations, and API improvements. This guide explains the written context that helps issues and pull requests move smoothly. 

If you want to add a new model adapter or change the embedder contract, start with [Extending](extending.md).



## GitHub Workflow

All development happens on GitHub. If you already know what needs to change, the fastest path is usually to open a pull request. If the scope is still unclear, or if the change affects public API shape, model semantics, or repository direction, opening an issue first helps everyone start from the same written context.

For normal code contributions, fork the repository, create a focused branch, make the change, and open the pull request against `main`. Keeping unrelated changes in separate branches and PRs makes review much easier and usually gets feedback back to you faster.

```bash
git clone https://github.com/<your-username>/rs-embed.git
cd rs-embed
git checkout -b <descriptive-branch-name>
```

When you open the issue or PR, please include direct links to the relevant docs page, paper, model card, upstream repository, or prior discussion. If your claim depends on behavior, a plain-text code snippet is especially helpful because reviewers can search it, run it, and quote it directly.

## Issues

If you are not sure how to write the report, this is usually enough:

````md
## Summary

What were you trying to do? What happened instead?

## Model

`prithvi`

## Backend

`auto`

## Minimal Reproduction

```python
from rs_embed import ...
...
```

## Setup

```python
spatial = ...
temporal = ...
output = ...
backend = ...
```

## Error

Paste the traceback or describe the incorrect result.

## Links

- docs:
- paper or model card:
- upstream repo:

## Environment

- Python:
- device:
- relevant versions:
````

The GitHub issue forms ask for the same information, but this template is a useful fallback when you want to draft the report first.

## Pull Requests

If you are not sure how to structure the pull request, this is usually enough:

````md
## Summary

What problem does this PR solve?

## Links

- issue:
- docs:
- upstream reference:

## Changes

What changed? What did you intentionally leave out?

## Verification

```bash
python -m pytest -v tests/...
```

Or paste a small code snippet that demonstrates the new behavior.

## Docs

What docs were updated? 

````

If the change modifies model behavior, it is especially helpful to describe the input contract, the defaults you chose, and any behavior that remains intentionally unsupported. If the change affects user-facing semantics, updating the docs in the same pull request is usually the right move.

## Tests

`rs-embed` uses GitHub Actions for continuous integration, and it helps a lot if pull requests keep those checks green. If you change Python behavior, please add or update tests so the new behavior is explicit and easy to review. The repository already contains many focused tests, so the easiest starting point is usually to find a nearby test and mirror its style rather than inventing a new pattern.

For a normal local setup, install the development dependencies first:

```bash
pip install -e ".[dev,full]"
```

Then run the checks that cover your change:

```bash
ruff check src/
ruff format src/ tests/
python -m pytest -v -n auto
```

If your change is local to one module or behavior, using a narrower pytest target is completely fine. Including that command in the pull request gives reviewers a concrete reproduction path instead of just a general statement that testing passed.

## Documentation

Documentation changes are real contributions, not just cleanup after the fact. If you change the public API, model semantics, installation story, or extension workflow, it helps a lot to update the docs while the implementation context is still fresh.

For doc changes, build the site locally before opening the pull request:

```bash
pip install mkdocs-material pymdown-extensions
mkdocs build --strict
```

The repository also includes GitHub issue and pull request templates. They are there to make it easier to include the right context in the first message: links, code, assumptions, and verification. That shared written context is what keeps collaboration efficient and cuts down on unnecessary back-and-forth.
