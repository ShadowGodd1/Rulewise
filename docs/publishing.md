# Publishing

This page is the release checklist. The package name `rulewise` is available on PyPI as of
2026-11-12. The version is set in `pyproject.toml` and `src/rulewise/__init__.py`; keep the two
in sync.

## 1. Preflight

```bash
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
```

`twine check` validates the long description and the metadata. If it reports unknown fields such
as `license-expression` or `license-file`, upgrade `packaging` to 24.2 or newer, or upgrade
twine.

Check the version and the name one more time:

```bash
python -c "import rulewise; print(rulewise.__version__)"
```

A version that already exists on PyPI or TestPyPI cannot be replaced. Bump it before uploading.

## 2. Build

From a clean tree:

```bash
rm -rf build dist src/*.egg-info
python -m build --sdist --wheel
python -m twine check dist/*
```

The wheel should contain only `rulewise` and its metadata. The source distribution also contains
the tests, the benchmarks, the docs and the changelog. Public datasets are not bundled; users run
`python benchmarks/get_data.py` if they want to reproduce the benchmarks.

## 3. Upload to TestPyPI first

Create a TestPyPI account and an API token, then:

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a clean environment:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ rulewise
python -c "import rulewise; print(rulewise.__version__)"
```

Use the extra index so that numpy, scipy and pandas come from PyPI.

## 4. Upload to PyPI

Create a PyPI API token scoped to the `rulewise` project. Use `__token__` as the username:

```bash
python -m twine upload dist/*
```

Twine reads `TWINE_USERNAME` and `TWINE_PASSWORD` from the environment, or from `~/.pypirc`, or
from the keyring. Never commit a token and never pass it on a command line that is recorded in
shell history. A typical local flow is:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
python -m twine upload dist/*
```

On Windows PowerShell:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-..."
python -m twine upload dist/*
```

## 5. Preferred: trusted publishing from GitHub Actions

The repository includes `.github/workflows/publish.yml`. It builds the distributions, checks
them, and publishes with OIDC trusted publishing when a GitHub release is published. No PyPI
token is stored in the repository.

One-time setup on PyPI:

1. Open the project on PyPI, then Publishing, then Add a trusted publisher.
2. Choose GitHub.
3. Enter owner `ShadowGodd1`, repository `Rulewise`, workflow filename `publish.yml`, and
   environment name `pypi`.
4. Save.

Then create a GitHub release with a tag such as `v0.1.0`. The workflow publishes the version in
`pyproject.toml`; the tag is only a label.

If you prefer a token-based workflow, add `PYPI_API_TOKEN` as a repository secret and pass it to
`pypa/gh-action-pypi-publish` with `password: ${{ secrets.PYPI_API_TOKEN }}`.

## 6. After upload

* Check the PyPI page: README rendering, classifiers, project links, licence and Python
  requirement.
* Install the released wheel in a clean environment and run the five-minute example.
* Tag the release in version control.
* Update `CHANGELOG.md` with the release date and move unreleased items under the version.
* Build and deploy the documentation site if you maintain one.

## Documentation site

The docs are plain Markdown with a `mkdocs.yml` configuration. The `docs` workflow builds and
pushes them to the `gh-pages` branch. Enable GitHub Pages once in the repository settings:

1. Open Settings, then Pages.
2. Under Build and deployment, choose Deploy from a branch.
3. Select the `gh-pages` branch and the root folder.

The site is then served at `https://shadowgodd1.github.io/Rulewise/`. To build locally:

```bash
pip install mkdocs mkdocs-material
python -m mkdocs build --strict
python -m mkdocs serve
```

`python -m mkdocs build --strict` fails on broken internal links and missing pages. Run it before
a release if the documentation site is part of the announcement. Newer versions of
`mkdocs-material` print an end-of-life notice that strict mode treats as a warning; set
`NO_MKDOCS_2_WARNING=1` in CI if you see that message.

## Version policy

Use semantic versioning. Before 1.0, minor releases may change behaviour. A release that changes
the default mode, the correction, the output columns or the meaning of an existing parameter is a
breaking change and needs a minor version bump with a migration note in `CHANGELOG.md`.

## Checklist

- [ ] Tests pass: `python -m pytest -q`
- [ ] Version is correct in `pyproject.toml` and `src/rulewise/__init__.py`
- [ ] `CHANGELOG.md` has an entry for the version
- [ ] `python -m build` succeeds
- [ ] `python -m twine check dist/*` passes
- [ ] TestPyPI install works
- [ ] PyPI upload uses a scoped token or trusted publishing
- [ ] Clean install from PyPI runs the five-minute example
- [ ] Documentation builds with `mkdocs build --strict`
