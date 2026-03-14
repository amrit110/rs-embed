# Release Checklist

This file is the practical checklist for publishing `rs-embed` to PyPI.

## Release Checklist

### 1. Version and metadata

- [ ] Bump version in [src/rs_embed/_version.py](src/rs_embed/_version.py).
- [ ] Confirm README install instructions still match the intended release path.
- [ ] Confirm project URLs, license, base runtime dependencies, and extras (`gee`, `gee-auth`, `models`, `full`) in [pyproject.toml](pyproject.toml) are correct.

### 2. Clean environment validation

- [ ] Create a fresh Python 3.12 environment.
- [ ] Upgrade packaging tools:

```bash
python -m pip install --upgrade pip build twine
```

- [ ] Install dev/test dependencies:

```bash
pip install -e ".[dev,full]"
```

- [ ] Run the test suite:

```bash
pytest -q
```

Notes:

- CI currently excludes GEE-authenticated integration testing.
- If you want to smoke-test GEE flows manually, do that separately after `earthengine authenticate`.

### 3. Build artifacts

- [ ] Remove old artifacts:

```bash
rm -rf dist build *.egg-info
```

- [ ] Build both wheel and sdist:

```bash
python -m build
```

- [ ] Validate package metadata and README rendering:

```bash
python -m twine check dist/*
```

- [ ] Inspect artifact contents if anything looks suspicious:

```bash
python -m zipfile -l dist/rs_embed-*.whl
tar -tzf dist/rs_embed-*.tar.gz | head -200
```

### 4. Installability smoke tests

- [ ] In a fresh environment, install the built wheel:

```bash
python -m venv /tmp/rs-embed-release-check
/tmp/rs-embed-release-check/bin/pip install dist/rs_embed-*.whl
```

- [ ] Verify import and CLI:

```bash
/tmp/rs-embed-release-check/bin/python -c "import rs_embed; print(rs_embed.__version__)"
/tmp/rs-embed-release-check/bin/rs-embed --help
```

- [ ] Smoke-test the standard install path:

```bash
/tmp/rs-embed-release-check/bin/pip install \
  "rs-embed[full] @ file://$PWD/dist/rs_embed-<version>-py3-none-any.whl"
```

### 5. Git state

- [ ] Ensure the working tree is clean.
- [ ] Commit release changes.
- [ ] Tag the release:

```bash
git tag v<version>
git push origin main --tags
```

### 6. Publish

Recommended first pass:

- [ ] Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

- [ ] Test installation from TestPyPI in a fresh environment.

Production publish:

- [ ] Upload to PyPI:

```bash
python -m twine upload dist/*
```

### 7. Post-release verification

- [ ] Confirm the package page on PyPI renders correctly.
- [ ] Install from PyPI in a fresh environment:

```bash
python -m venv /tmp/rs-embed-pypi-check
/tmp/rs-embed-pypi-check/bin/pip install rs-embed
/tmp/rs-embed-pypi-check/bin/python -c "import rs_embed; print(rs_embed.__version__)"
```

- [ ] Create the GitHub release notes for the tag.

## Project-specific release notes

These are easy to miss in `rs-embed`:

- Version is sourced from [src/rs_embed/_version.py](src/rs_embed/_version.py) via Hatch; there is no second version field to keep in sync.
- `README.md` should use absolute GitHub/raw URLs for images and notebook links so PyPI renders correctly.
- `import rs_embed` should remain lightweight enough that missing optional model deps do not break the core install path.
- The core dependency set should stay conservative; avoid widening `numpy` without checking downstream `xarray/pandas/scipy` compatibility.
- The `rs-embed` console entry point should keep working after wheel install.
- `rs-embed[gee]` should stay lighter than `rs-embed[gee-auth]`; `rs-embed[models]` should remain the practical install target for on-the-fly model runtimes.
- `rs-embed[full]` is the practical one-command install target; keeping everything in base dependencies causes heavy solver/backtracking pressure.
- GEE-backed workflows still require `earthengine authenticate` after install.
- `THOR` still requires `thor_terratorch_ext`, which is outside the standard PyPI dependency set.
