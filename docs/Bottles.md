# Bottles (Binary Packages)

Bottles are produced by installing a formula with `brew install --build-bottle <formula>` and then bottling it with `brew bottle <formula>`. This outputs the bottle DSL which should be inserted into the formula file.

## Usage
If a bottle is available and usable it will be downloaded and poured automatically when you `brew install <formula>`. If you wish to disable this you can do so by specifying `--build-from-source`.

Bottles will not be used if the user requests it (see above), if the formula requests it (with `pour_bottle?`), if any options are specified during installation (bottles are all compiled with default options), if the bottle is not up to date (e.g. lacking a checksum) or if the bottle's `cellar` is not `:any` nor equal to the current `HOMEBREW_CELLAR`.

## Creation
Bottles are created using the [Brew Test Bot](Brew-Test-Bot.md), usually when people submit pull requests to Homebrew. The `bottle do` block is updated by maintainers when they merge a pull request. For the Homebrew organisations' taps they are uploaded to and downloaded from [Bintray](https://bintray.com/homebrew).

By default, bottles will be built for the oldest CPU supported by the OS/architecture you're building for (Core 2 for 64-bit OSs). This ensures that bottles are compatible with all computers you might distribute them to. If you *really* want your bottles to be optimised for something else, you can pass the `--bottle-arch=` option to build for another architecture; for example, `brew install foo --build-bottle --bottle-arch=penryn`. Just remember that if you build for a newer architecture some of your users might get binaries they can't run and that would be sad!

## Format
Bottles are simple gzipped tarballs of compiled binaries. Any metadata is stored in a formula's bottle DSL and in the bottle filename (i.e. macOS version, revision).

## Bottle DSL (Domain Specific Language)
Bottles have a DSL to be used in formulae which is contained in the `bottle do ... end` block.

A simple (and typical) example:

```ruby
bottle do
  sha256 arm64_big_sur: "a9ae578b05c3da46cedc07dd428d94a856aeae7f3ef80a0f405bf89b8cde893a"
  sha256 big_sur:       "5dc376aa20241233b76e2ec2c1d4e862443a0250916b2838a1ff871e8a6dc2c5"
  sha256 catalina:      "924afbbc16549d8c2b80544fd03104ff8c17a4b1460238e3ed17a1313391a2af"
  sha256 mojave:        "678d338adc7d6e8c352800fe03fc56660c796bd6da23eda2b1411fed18bd0d8d"
end
```

A full example:

```ruby
bottle do
  root_url "https://example.com"
  rebuild 4
  sha256 cellar: "/opt/homebrew/Cellar", arm64_big_sur: "a9ae578b05c3da46cedc07dd428d94a856aeae7f3ef80a0f405bf89b8cde893a"
  sha256 cellar: :any,                   big_sur:       "5dc376aa20241233b76e2ec2c1d4e862443a0250916b2838a1ff871e8a6dc2c5"
  sha256                                 catalina:      "924afbbc16549d8c2b80544fd03104ff8c17a4b1460238e3ed17a1313391a2af"
  sha256                                 mojave:        "678d338adc7d6e8c352800fe03fc56660c796bd6da23eda2b1411fed18bd0d8d"
end
```

### Root URL (`root_url`)
Optionally contains the URL root used to calculate bottle URLs.
By default this is omitted and the Homebrew default bottle URL root is used. This may be useful for taps which wish to provide bottles for their formulae or to cater for a non-default `HOMEBREW_CELLAR`.

### Cellar (`cellar`)
Optionally contains the value of `HOMEBREW_CELLAR` in which the bottles were built.
Most compiled software contains references to its compiled location so cannot be simply relocated anywhere on disk. If this value is `:any` or `:any_skip_relocation` this means that the bottle can be safely installed in any Cellar as it did not contain any references to its installation Cellar. This can be omitted if a bottle is compiled (as all default Homebrew ones are) for the default `HOMEBREW_CELLAR` of `/usr/local/Cellar`.

### Rebuild version (`rebuild`)
Optionally contains the rebuild version of the bottle.
Sometimes bottles may need be updated without bumping the version of the formula, e.g. a new patch was applied. In that case the rebuild will have a value of 1 or more.

### Checksum (`sha256`)
Contains the SHA-256 hash of a bottle for a particular version of macOS.

## Formula DSL
An additional method is available in the formula DSL.

### Pour bottle (`pour_bottle?`)
Optionally returns a boolean to decide whether a bottle should be used for this formula.
For example a bottle may break if another formula has been compiled with non-default options, so this method could check for that case and return `false`.

A full example:

```ruby
pour_bottle? do
  reason "The bottle needs the Xcode CLT to be installed."
  satisfy { MacOS::CLT.installed? }
end
```
