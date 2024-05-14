# Homebrew Docs

These are the source files for the [Homebrew documentation site](https://docs.brew.sh/).

A [GitHub Action](https://github.com/Homebrew/brew/blob/master/.github/workflows/docs.yml) is run to validate each change before the site is deployed to GitHub Pages.

## Usage

Open <https://docs.brew.sh> in your web browser.

To instead generate the site locally to <http://localhost:4000>, run:

```bash
cd `brew --repository`/docs
bundle install
bundle exec jekyll serve --watch
```
