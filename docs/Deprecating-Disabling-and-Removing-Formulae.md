# Deprecating, Disabling, and Removing Formulae

There are many reasons why formulae may be deprecated, disabled, or removed. This document explains the differences between each method as well as explaining when one method should be used over another.

## Overview

This general rule of thumb can be followed:

- `deprecate!` should be used for formulae that _should_ no longer be used.
- `disable!` should be used for formulae that _cannot_ be used.
- Formulae that are no longer acceptable in homebrew/core or have been disabled for over a year should be removed.

## Deprecation

If a user attempts to install a deprecated formula, they will be shown a warning message but the install will proceed.

A formula should be deprecated to indicate to users that the formula should not be used and will be disabled in the future. Deprecated formulae should be maintained by the Homebrew maintainers so they can still build from source and their bottles continue to work (even if unmaintained upstream). If this is not possible, they should be disabled.

The most common reasons for deprecation are when the upstream project is deprecated, unmaintained, or archived.

Formulae with dependents may be deprecated only if at least one of the following are true:

- its dependents are all deprecated
- the formula does not build on any of our supported macOS versions and on Linux
- the formula has outstanding CVEs

To deprecate a formula, add a `deprecate!` call. This call should include a deprecation date (in the ISO 8601 format) and a deprecation reason:

```ruby
deprecate! date: "YYYY-MM-DD", because: :reason
```

The `date` parameter should be set to the date that the project or version became (or will become) deprecated. If there is no clear date but the formula needs to be deprecated, use today's date. If the `date` parameter is set to a date in the future, the formula will not become deprecated until that date. This can be useful if the upstream developers have indicated a date where the project or version will stop being supported.

The `because` parameter can be a preset reason (using a symbol) or a custom reason. See the [Deprecate and Disable Reasons](#deprecate-and-disable-reasons) section below for more details about the `because` parameter.

## Disabling

If a user attempts to install a disabled formula, they will be shown an error message and the install will fail.

A formula should be disabled to indicate to users that the formula cannot be used and will be removed in the future. Disabled formulae may no longer build from source or have working bottles.

The most common reasons for disabling a formula are:

- it cannot be built from source (meaning no bottles can be built)
- it has been deprecated for a long time
- the upstream repository has been removed
- the project has no license

Formulae should not be disabled without a deprecation period of at least three months unless the circumstances are exceptional (e.g. the formula does not build on any supported macOS version or Linux). Popular formulae should have longer deprecation periods. The popularity of a formula should be based on our analytics data.

**Note: disabled formulae in homebrew/core will be automatically removed one year after their disable date**

To disable a formula, add a `disable!` call. This call should include a deprecation date (in the ISO 8601 format) and a deprecation reason:

```ruby
disable! date: "YYYY-MM-DD", because: :reason
```

The `date` parameter should be set to the date that the reason for disabling came into effect. If there is no clear date but the formula needs to be disabled, use today's date. If the `date` parameter is set to a date in the future, the formula will be deprecated until that date (on which the formula will become disabled).

The `because` parameter can be a preset reason (using a symbol) or a custom reason. See the [Deprecate and Disable Reasons](#deprecate-and-disable-reasons) section below for more details about the `because` parameter.

## Removal

A formula should be removed if it does not meet our criteria for [acceptable formulae](Acceptable-Formulae.md) or [versioned formulae](Versions.md), has a non-open-source license, or has been disabled for over a year.

## Deprecate and Disable Reasons

When a formula is deprecated or disabled, a reason explaining the action must be provided.

There are two ways to indicate the reason. The preferred way is to use a pre-existing symbol to indicate the reason. The available symbols are listed below and can be found in the [`DeprecateDisable` module](https://github.com/Homebrew/brew/blob/master/Library/Homebrew/deprecate_disable.rb):

- `:does_not_build`: the formula cannot be built from source
- `:no_license`: the formula does not have a license
- `:repo_archived`: the upstream repository has been archived
- `:repo_removed`: the upstream repository has been removed
- `:unmaintained`: the project appears to be abandoned
- `:unsupported`: Homebrew's application of the software is not supported by the upstream developers (e.g. upstream only supports macOS versions prior to 10.14)
- `:deprecated_upstream`: the project is deprecated upstream
- `:versioned_formula`: the formula is a versioned formula

These reasons can be specified by their symbols (the comments show the message that will be displayed to users):

```ruby
# Warning: <formula> has been deprecated because it is deprecated upstream!
deprecate! date: "2020-01-01", because: :deprecated_upstream
```

```ruby
# Error: <formula> has been disabled because it does not build!
disable! date: "2020-01-01", because: :does_not_build
```

If these pre-existing reasons do not fit, a custom reason can be specified. These reasons should be written to fit into the sentence `<formula> has been deprecated/disabled because it <reason>!`.

A well-worded example of a custom reason would be:

```ruby
# Warning: <formula> has been deprecated because it fetches unversioned dependencies at runtime!
deprecate! date: "2020-01-01", because: "fetches unversioned dependencies at runtime"
```

A poorly-worded example of a custom reason would be:

```ruby
# Error: <formula> has been disabled because it invalid license!
disable! date: "2020-01-01", because: "invalid license"
```
