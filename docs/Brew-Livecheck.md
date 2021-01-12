# `brew livecheck`

The `brew livecheck` command finds the newest version of a formula or cask's software by checking upstream. Livecheck has [strategies](https://rubydoc.brew.sh/Homebrew/Livecheck/Strategy.html) to identify versions from various sources, such as Git repositories, websites, etc.

## Behavior

When livecheck isn't given instructions for how to check for upstream versions, it does the following by default:

1. For formulae: Collect the `head`, `stable`, and `homepage` URLs, in that order. For casks: Collect the `url` and `homepage` URLs, in that order.
1. Determine if any strategies apply to the first URL. If not, try the next URL.
1. If a strategy can be applied, use it to check for new versions.
1. Return the newest version (or an error if versions could not be found at any available URLs).

It's sometimes necessary to override this default behavior to create a working check for a formula/cask. If a source doesn't provide the newest version, we need to check a different one. If livecheck doesn't correctly match version text, we need to provide an appropriate regex.

This can be accomplished by adding a `livecheck` block to the formula/cask. For more information on the available methods, please refer to the [`Livecheck` class documentation](https://rubydoc.brew.sh/Livecheck.html).

## Creating a check

1. **Use the debug output to understand the situation**. `brew livecheck --debug <formula>|<cask>` provides information about which URLs livecheck tries, any strategies that apply, matched versions, etc.

1. **Research available sources to select a URL**. Try removing the file name from `stable`/`url`, to see if this is a directory listing page. If that doesn't work, try to find a page that links to the file (e.g. a download page). If it's not possible to find the newest version on the website, try checking other sources from the formula/cask. When necessary, search for other sources outside of the formula/cask.

1. **Create a regex, if necessary**. If the check works without a regex and wouldn't benefit from having one, it's usually fine to omit it. More information on creating regexes can be found in the [regex guidelines](#regex-guidelines) section.

### General guidelines

* **Only use `strategy` when it's necessary**. For example, if livecheck is already using `Git` for a URL, it's not necessary to use `strategy :git`. However, if `Git` applies to a URL but we need to use `PageMatch`, it's necessary to use `strategy :page_match`.

* **Only use the `GithubLatest` strategy when it's necessary and correct**. Github.com rate limits requests and we try to minimize our use of this strategy to avoid hitting the rate limit on CI or when using `brew livecheck --tap` on large taps (e.g. homebrew/core). The `Git` strategy is often sufficient and we only need to use `GithubLatest` when the "latest" release is different than the newest version from the tags.

### URL guidelines

* **A `url` is required in a `livecheck` block**. This can be a URL string (e.g. `"https://www.example.com/downloads/"`) or a formula/cask URL symbol (i.e. `:stable`, `:url`, `:head`, `:homepage`). The exception to this rule is a `livecheck` block that only uses `skip`.

* **Check for versions in the same location as the stable archive, whenever possible**.

* **Avoid checking paginated release pages, when possible**. For example, we generally avoid checking the `release` page for a GitHub project because the latest stable version can be pushed off the first page by pre-release versions. In this scenario, it's more reliable to use the `Git` strategy, which fetches all the tags in the repository.

### Regex guidelines

The `livecheck` block regex restricts matches to a subset of the fetched content and uses a capture group around the version text.

* **Regexes should be made case insensitive, whenever possible**, by adding `i` at the end (e.g. `/.../i` or `%r{...}i`). This improves reliability, as the regex will handle changes in letter case without needing modifications.

* **Regexes should only use a capturing group around the version text**. For example, in `/href=.*?example-v?(\d+(?:\.\d+)+)(?:-src)?\.t/i`, we're only using a capturing group around the version test (matching a version like `1.2`, `1.2.3`, etc.) and we're using non-capturing groups elsewhere (e.g. `(?:-src)?`).

* **Anchor the start/end of the regex, to restrict the scope**. For example, on HTML pages we often match file names or version directories in `href` attribute URLs (e.g. `/href=.*?example[._-]v?(\d+(?:\.\d+)+)\.zip/i`). The general idea is that limiting scope will help exclude unwanted matches.

* **Avoid generic catch-alls like `.*` or `.+`** in favor of something non-greedy and/or contextually appropriate. For example, to match characters within the bounds of an HTML attribute, use `[^"' >]+?`.

* **Use `[._-]` in place of a period/underscore/hyphen between the software name and version in a file name**. For a file named `example-1.2.3.tar.gz`, `example[._-]v?(\d+(?:\.\d+)+)\.t` will continue matching if the upstream file name format changes to `example_1.2.3.tar.gz` or `example.1.2.3.tar.gz`.

* **Use `\.t` in place of `\.tgz`, `\.tar\.gz`, etc.** There are a variety of different file extensions for tarballs (e.g. `.tar.bz2`, `tbz2`, `.tar.gz`, `.tgz`, `.tar.xz`, `.txz`, etc.) and the upstream source may switch from one compression format to another over time. `\.t` avoids this issue by matching current and future formats starting with `t`. Outside of tarballs, we use the full file extension in the regex like `\.zip`, `\.jar`, etc.

## Example `livecheck` blocks

The following examples cover a number of patterns that you may encounter. These are intended to be representative samples and can be easily adapted.

When in doubt, start with one of these examples instead of copy-pasting a `livecheck` block from a random formula/cask.

### File names

```ruby
  livecheck do
    url "https://www.example.com/downloads/"
    regex(/href=.*?example[._-]v?(\d+(?:\.\d+)+)\.t/i)
  end
```

When matching the version from a file name on an HTML page, we often restrict matching to `href` attributes. `href=.*?` will match the opening delimiter (`"`, `'`) as well as any part of the URL before the file name.

We sometimes make this more explicit to exclude unwanted matches. URLs with a preceding path can use `href=.*?/` and others can use `href=["']?`. For example, this is necessary when the page also contains unwanted files with a longer prefix (`another-example-1.2.tar.gz`).

### Version directories

```ruby
  livecheck do
    url "https://www.example.com/releases/example/"
    regex(%r{href=["']?v?(\d+(?:\.\d+)+)/?["' >]}i)
  end
```

When checking a directory listing page, sometimes files are separated into version directories (e.g. `1.2.3/`). In this case, we must identify versions from the directory names.

### Git tags

```ruby
  livecheck do
    url :stable
    regex(/^v?(\d+(?:\.\d+)+)$/i)
  end
```

When the `stable` URL uses the `Git` strategy, the regex above will only match tags like `1.2`/`v1.2`, etc.

If tags include the software name as a prefix (e.g. `example-1.2.3`), it's easy to modify the regex accordingly: `/^example[._-]v?(\d+(?:\.\d+)+)$/i`

### `PageMatch` `strategy` block

```ruby
  livecheck do
    url :homepage
    regex(/href=.*?example[._-]v?(\d{4}-\d{2}-\d{2})\.t/i)
    strategy :page_match do |page, regex|
      page.scan(regex).map { |match| match&.first&.gsub(/\D/, "") }
    end
  end
```

When necessary, a `strategy` block allows us to have greater flexibility in how upstream version information is matched and processed. Currently, they're only used when the upstream version format needs to be manipulated to match the formula/cask format. In the example above, we're converting a date format like `2020-01-01` into `20200101`.

The `PageMatch` `strategy` block style seen here also applies to any strategy that uses `PageMatch` internally.

### `Git` `strategy` block

```ruby
  livecheck do
    url :stable
    regex(/^(\d{4}-\d{2}-\d{2})$/i)
    strategy :git do |tags, regex|
      tags.map { |tag| tag[regex, 1]&.gsub(/\D/, "") }.compact
    end
  end
```

A `strategy` block for `Git` is a bit different, as the block receives an array of tag strings instead of a page content string. Similar to the `PageMatch` example, this is converting tags with a date format like `2020-01-01` into `20200101`.

### Skip

```ruby
  livecheck do
    skip "No version information available"
  end
```

Livecheck automatically skips some formulae/casks for a number of reasons (deprecated, disabled, discontinued, etc.). However, on rare occasions we need to use a `livecheck` block to do a manual skip. The `skip` method takes a string containing a very brief reason for skipping.
