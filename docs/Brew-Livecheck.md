# Brew Livecheck

**NOTE: This document is a work in progress and will be revised and expanded as time permits.**

**NOTE: `livecheck` blocks are currently found in separate files in the [Homebrew/homebrew-livecheck](https://github.com/Homebrew/homebrew-livecheck) repository's `Livecheckables` folder. These will be migrated to their respective formulae in Homebrew/homebrew-core in the near future and this document is written as if this migration has already happened.**

The general purpose of the `brew livecheck` command is to find the newest version of a formula's software by checking an upstream source. Livecheck has [built-in strategies](#built-in-strategies) that can identify versions from some popular sources, such as Git repositories, certain websites, etc.

## Default behavior

When livecheck isn't given instructions for how to check for upstream versions of a formula's software, it does the following by default:

1. Collect the `head`, `stable`, and `homepage` URLs from the formula, in that order.
2. Determine if any of the available strategies can be applied to the first URL. Move on to the next URL if no strategies apply.
3. If a strategy can be applied, use it to check for new versions.
4. Return the newest version (or an error if versions could not be found at any available URLs).

This approach works fine for a number of formulae without requiring any manual intervention. However, it's sometimes necessary to change livecheck's default behavior to create a working check for a formula.

It may be that the source livecheck is using doesn't provide the newest version and we need to check a different one instead. In another case, livecheck may be matching a version it shouldn't and we need to provide a regex to only match what's appropriate.

## The `livecheck` block

We can control livecheck's behavior by providing a `livecheck` block in the formula. Here is a simple example to check a "downloads" page for links containing a filename like `example-1.2.tar.gz`:

```ruby
livecheck do
  url "https://www.example.com/downloads/"
  regex(/href=.*?example[._-]v?(\d+(?:\.\d+)+)\.t/i)
end
```

At the moment, it's only necessary to create a `livecheck` block in a formula when the default check doesn't work properly.

## Creating a check

1. **Use the debug output to understand the current situation**. Running `brew livecheck --debug <formula>` (where `<formula>` is the formula name) will provide more information about which URL livecheck is using and any strategy that applies.

2. **Research available sources**. It's generally preferable to check for a new version at the same source as the `stable` URL, when possible. With this in mind, it may be a good idea to start by removing the file name from the `stable` URL, to see if this is a directory listing page. If that doesn't work, the website may have a downloads page we can check for versions. If it's not possible to find the newest version at this source through any means, try checking other sources from the formula (e.g. an upstream Git repository or the homepage). It's also sometimes necessary to search for other sources outside of the formula.

3. **Compare available versions between sources**. If the latest version is available from the `stable` source, it's best to use that. Otherwise, check the other sources to identify where the latest version is available.

4. **Select a source**. After researching and comparing sources, decide which one is the best available option and use it as the `url` in the `livecheck` block.

5. **Create a regex, if necessary or beneficial**. If the check works fine without a regex and wouldn't benefit from having one, it's fine to omit it. However, when a default check isn't working properly and we need to create a `livecheck` block, a regex is almost always necessary as well. More information on creating regexes can be found in the [regex guidelines](#regex-guidelines) section.

6. **Verify the check is working as intended**. Run `brew livecheck --debug <formula>` again to ensure livecheck is identifying all the versions it should and properly returning the newest version at the end.

### URL guidelines

* **A `url` is a required part of a `livecheck` block** and can either be a string containing a URL (e.g. `"https://www.example.com/downloads/"`) or a symbol referencing one of the supported formula URLs (i.e. `:stable`, `:homepage`, or `:head`).

* **Use a symbol for a formula URL (i.e. `:homepage`, `:stable`, and `:head`) when appropriate**, to avoid duplicating formula URLs in the livecheckable.

* **It's generally preferable to check for versions in the same location as the stable archive, when possible**. This preference is stronger for first-party sources (websites, repositories, etc.) and becomes weaker for third-party sources (e.g. mirrors, another software package manager, etc.).

### Regex guidelines

The regex in a `livecheck` block is used within a strategy to restrict matching to only relevant text (or strings) and to establish what part of the matched text is the version (using a capture group).

Creating a good regex is a balance between being too strict (breaking easily) and too loose (matching more than it should).

* For technical reasons, **the `regex` call in the `livecheck` block should always use parentheses** (e.g. `regex(/example/)`).

* **Regex literals should follow the established Homebrew style**, where the `/.../` syntax is the default and the `%r{...}` syntax is used when a forward slash (`/`) is present.

* **Regexes should adequately represent their intention.** This requires an understanding of basic regex syntax, so we avoid issues like using `.` (match any character) instead of `\.` when we only want to match a period. We also try to be careful about our use of generic catch-alls like  `.*` or `.+`, as it's often better to use something non-greedy and contextually appropriate. For example, if we wanted to match a variety of characters while trying to stay within the bounds of an HTML attribute, we could use something like `[^"' >]*?`.

* **Try not to be too specific in some parts of the regex.** For example, if a file name uses a hyphen (`-`) between the software name and version (e.g. `example-1.2.3.tar.gz`), we may want to use something like `example[._-]v?(\d+(?:\.\d+)+)\.t` instead. This would allow the regex to continue matching if the upstream file name format changes to `example.1.2.3.tar.gz` or `example_1.2.3.tar.gz`.

* **Regexes should be case insensitive unless case sensitivity is explicitly required for matching to work properly**. Case insensitivity is enabled by adding the `i` flag at the end of the regex literal (e.g. `/.../i` or `%r{...}i`). This helps to improve reliability and reduce maintenance, as a case-insensitive regex doesn't need to be manually updated if there are any upstream changes in letter case.

* **Regexes should only use a capturing group around the part of the matched text that corresponds to the version**. For example, in `/href=.*?example-v?(\d+(?:\.\d+)+)(?:-src)?\.t/i`, we're only using a capturing group around the version part (matching a version like `1.2`, `1.2.3`, etc.) and we're using non-capturing groups elsewhere (e.g. `(?:-src)?`). This allows livecheck to rely on the first capture group being the version string.

* **Regexes should only match stable versions**. Regexes should be written to avoid prerelease versions like `1.2-alpha1`, `1.2-beta1`, `1.2-rc1`, etc.

* **Restrict matching to `href` attributes when targeting file names in an HTML page (or `url` attributes in an RSS feed)**. Using `href=.*?` (or `url=.*?`) at the start of the regex will take care of any opening delimiter for the attribute (`"`, `'`, or nothing) as well as any leading part of the URL. This helps to keep the regex from being overly specific, reducing the need for maintenance in the future. A regex like `href=.*?example-...` is often fine but sometimes it's necessary to have something explicit before the file name to limit matching to only what's appropriate (e.g. `href=.*?/example-...` or `href=["']?example-...`). Similarly, `["' >]` can be used to target the end of the attribute, when needed.

* **Use `\.t` in place of `\.tgz`, `\.tar\.gz`, etc.** There are a number of different file extensions for tarballs (e.g. `.tar.bz2`, `tbz2`, `.tar.gz`, `.tgz`, `.tar.xz`, `.txz`, etc.) and the upstream source may switch from one compression format to another over time. `\.t` avoids this issue by matching current and future formats starting with `t`. Outside of tarballs, it's fine to use full file extensions in the regex like `\.zip`, `\.jar`, etc.

* **When matching versions like `1.2`, `1.2.3`, `v1.2`, etc., use the standard snippet for this in the regex: `v?(\d+(?:\.\d+)+)`**. This is often copy-pasted into the regex but it can also be modified to suit the circumstances. For example, if the version uses underscores instead, the standard regex could be modified to something like `v?(\d+(?:[._]\d+)+)`. [The general idea behind this standard snippet is that it better represents our intention compared to older, looser snippets that we now avoid (e.g. `[0-9.]+`).]

* **Similarly, when matching Git tags with a version like `1.2`, `1.2.3`, `v1.2.3`, etc., start with the standard regex for this (`/^v?(\d+(?:\.\d+)+)$/i`) and modify it as needed**. Sometimes it's necessary to modify the regex to add a prefix, like `/^example-v?(\d+(?:\.\d+)+)$/i` for an `example-1.2.3` tag format. The general idea here is that Git tags are strings, so we can avoid unrelated software by restricting the start of the string (`^`) and unstable versions by restricting the end of the string (`$`).

## Built-in strategies

Livecheck's strategies are established methods for finding versions at either a specific source or a general type of source. The available strategies are as follows:

* `Apache`
* `Bitbucket`
* `Git`
* `Gnome`
* `Gnu`
* `Hackage`
* `Launchpad`
* `Npm`
* `PageMatch`
* `Pypi`
* `Sourceforge`

Each strategy has a `#match?(url)` method which determines whether the strategy can be applied to the provided URL. The `PageMatch` strategy is used as a fallback when a regex is provided and no other strategies apply. `PageMatch` simply uses the regex to match content on a page, so it's the desired strategy for URLs where a more-specific strategy doesn't apply.

Some of the strategies generate a URL and regex internally. In these cases, the strategy often derives information from the provided URL and uses it to create the URL it will check and the regex used for matching. However, if a `regex` is provided in the `livecheck` block, it will be used instead of any generated regex.

Livecheck also has a simple numeric priority system, where 5 is the default unless a strategy has defined its own `PRIORITY` constant. Currently, the `Git` strategy has a higher priority (8) and the `PageMatch` strategy has a low priority (0). In practice, this means that when more than one strategy applies to a URL (usually a specific strategy and `PageMatch`), the higher priority strategy is the one that's used.

### Tap strategies

Taps can add strategies to apply to their formulae by creating a `livecheck_strategy` folder in the root directory and placing strategy files within. At a minimum, strategies must provide a `#match?(url)` method and a `#find_versions(url, regex)` method.

The `#match?(url)` method takes a URL string and returns `true` or `false` to indicate whether the strategy can be applied to the URL.

`#find_versions(url, regex)` takes a URL and an optional regex and returns a `Hash` with a format like `{ :matches => {}, :regex => regex, :url => url }`. The `:matches` `Hash` uses version strings as the keys (e.g. `"1.2.3"`) and `Version` objects as the values. `:regex` is either the strategy-generated regex (if applicable), the regex provided as an argument, or `nil`. The `:url` is either the strategy-generated URL (if applicable) or the original URL provided.

The built-in strategies in Homebrew's `livecheck_strategy` folder may serve as examples to follow when creating tap strategies. Many of the built-in strategies simply generate a URL and regex before using the `PageMatch` strategy to do the heavy lifting (e.g. `PageMatch.find_versions(page_url, regex)`). When a strategy is checking a text page of some sort (e.g. HTML, RSS, etc.), it may be able to do the same thing. If a strategy needs to do something more complex, the `Git` and `PageMatch` strategies can be referenced as standalone examples.
