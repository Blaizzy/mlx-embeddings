# Cask Cookbook

Each Cask is a Ruby block, beginning with a special header line. The Cask definition itself is always enclosed in a `do … end` block. Example:

```ruby
cask "alfred" do
  version "2.7.1_387"
  sha256 "a3738d0513d736918a6d71535ef3d85dd184af267c05698e49ac4c6b48f38e17"

  url "https://cachefly.alfredapp.com/Alfred_#{version}.zip"
  name "Alfred"
  desc "Application launcher and productivity software"
  homepage "https://www.alfredapp.com/"

  app "Alfred 2.app"
  app "Alfred 2.app/Contents/Preferences/Alfred Preferences.app"
end
```

## The Cask Language Is Declarative

Each Cask contains a series of stanzas (or “fields”) which *declare* how the software is to be obtained and installed. In a declarative language, the author does not need to worry about **order**. As long as all the needed fields are present, Homebrew Cask will figure out what needs to be done at install time.

To make maintenance easier, the most-frequently-updated stanzas are usually placed at the top. But that’s a convention, not a rule.

Exception: `do` blocks such as `postflight` may enclose a block of pure Ruby code. Lines within that block follow a procedural (order-dependent) paradigm.

## Conditional Statements

### Efficiency

Conditional statements are permitted, but only if they are very efficient.
Tests on the following values are known to be acceptable:

| value                       | examples
| ----------------------------|--------------------------------------
| `MacOS.version`             | [coconutbattery.rb](https://github.com/Homebrew/homebrew-cask/blob/a11ee55e8ed8255f7dab77120dfb1fb955789559/Casks/coconutbattery.rb#L2-L16), [yasu.rb](https://github.com/Homebrew/homebrew-cask/blob/21d3f7ac8a4adac0fe474b3d4b020d284eeef88d/Casks/yasu.rb#L2-L23)

### Version Comparisons

Tests against `MacOS.version` may use either symbolic names or version
strings with numeric comparison operators:

```ruby
if MacOS.version <= :mojave        # symbolic name
```

```ruby
if MacOS.version <= "10.14"        # version string
```

The available symbols for macOS versions are: `:yosemite`, `:el_capitan`, `:sierra`, `:high_sierra`, `:mojave`, `:catalina` and `:big_sur`. The corresponding numeric version strings should be given as major releases containing a single dot.

Note that in the official Homebrew Cask repositories only the symbolic names are allowed. The numeric comparison may only be used for third-party taps.

### Always Fall Through to the Newest Case

Conditionals should be constructed so that the default is the newest OS version. When using an `if` statement, test for older versions, and then let the `else` statement hold the latest and greatest. This makes it more likely that the Cask will work without alteration when a new OS is released. Example (from [coconutbattery.rb](https://github.com/Homebrew/homebrew-cask/blob/2c801af44be29fff7f3cb2996455fce5dd95d1cc/Casks/coconutbattery.rb)):

```ruby
if MacOS.version <= :sierra
  # ...
elsif MacOS.version <= :mojave
  # ...
else
  # ...
end
```

### Switch Between Languages or Regions

If a cask is available in multiple languages, you can use the `language` stanza to switch between languages or regions based on the system locale.

## Arbitrary Ruby Methods

In the exceptional case that the Cask DSL is insufficient, it is possible to define arbitrary Ruby variables and methods inside the Cask by creating a `Utils` namespace. Example:

```ruby
cask "myapp" do
  module Utils
    def self.arbitrary_method
      ...
    end
  end

  name "MyApp"
  version "1.0"
  sha256 "a32565cdb1673f4071593d4cc9e1c26bc884218b62fef8abc450daa47ba8fa92"

  url "https://#{Utils.arbitrary_method}"
  homepage "https://www.example.com/"
  ...
end
```

This should be used sparingly: any method which is needed by two or more Casks should instead be rolled into the core. Care must also be taken that such methods be very efficient.

Variables and methods should not be defined outside the `Utils` namespace, as they may collide with Homebrew Cask internals.

## Header Line Details

The first non-comment line in a Cask follows the form:

```ruby
cask "<cask-token>" do
```

[`<cask-token>`](#token-reference) should match the Cask filename, without the `.rb` extension,
enclosed in single quotes.

There are currently some arbitrary limitations on Cask tokens which are in the process of being removed. The Travis bot will catch any errors during the transition.

## Stanza order

Having a common order for stanzas makes Casks easier to update and parse. Below is the complete stanza sequence (no Cask will have all stanzas). The empty lines shown here are also important, as they help to visually delineate information.

```
version
sha256

language

url
appcast
name
desc
homepage

livecheck

auto_updates
conflicts_with
depends_on
container

suite
app
pkg
installer
binary
manpage
colorpicker
dictionary
font
input_method
internet_plugin
prefpane
qlplugin
mdimporter
screen_saver
service
audio_unit_plugin
vst_plugin
vst3_plugin
artifact, target: # target: shown here as is required with `artifact`
stage_only

preflight

postflight

uninstall_preflight

uninstall_postflight

uninstall

zap

caveats
```

Note that every stanza that has additional parameters (`:symbols` after a `,`) shall have them on separate lines, one per line, in alphabetical order. An exception is `target:` which typically consists of short lines.

## Stanzas
### Required Stanzas

Each of the following stanzas is required for every Cask.

| name       | multiple occurrences allowed? | value                           |
| ---------- |------------------------------ | ------------------------------- |
| `version`  | no                            | Application version.<br />See [Version Stanza Details](#stanza-version) for more information.
| `sha256`   | no                            | SHA-256 checksum of the file downloaded from `url`, calculated by the command `shasum -a 256 <file>`. Can be suppressed by using the special value `:no_check`.<br />See [Checksum Stanza Details](#stanza-sha256) for more information.
| `url`      | no                            | URL to the `.dmg`/`.zip`/`.tgz`/`.tbz2` file that contains the application.<br />A [comment](#when-url-and-homepage-hostnames-differ-add-a-comment) should be added if the hostnames in the `url` and `homepage` stanzas differ. Block syntax should be used for URLs that change on every visit.<br />See [URL Stanza Details](#stanza-url) for more information.
| `name`     | yes                           | String providing the full and proper name defined by the vendor.<br />See [Name Stanza Details](#stanza-name) for more information.
| `desc`     | no                            | One-line description of the Cask. Shows when running `brew info`.<br />See [Desc Stanza Details](#stanza-desc) for more information.
| `homepage` | no                            | Application homepage; used for the `brew home` command.

### At Least One Artifact Stanza Is Also Required

Each Cask must declare one or more *artifacts* (i.e. something to install).

| name                | multiple occurrences allowed? | value                  |
| ------------------- |------------------------------ | ---------------------- |
| `app`               | yes                           | Relative path to an `.app` that should be moved into the `/Applications` folder on installation.<br />See [App Stanza Details](#stanza-app) for more information.
| `pkg`               | yes                           | Relative path to a `.pkg` file containing the distribution.<br />See [Pkg Stanza Details](#stanza-pkg) for more information.
| `binary`            | yes                           | Relative path to a Binary that should be linked into the `$(brew --prefix)/bin` folder (typically `/usr/local/bin`) on installation.<br />See [Binary Stanza Details](#stanza-binary) for more information.
| `colorpicker`       | yes                           | Relative path to a ColorPicker plugin that should be moved into the `~/Library/ColorPickers` folder on installation.
| `dictionary`        | yes                           | Relative path to a Dictionary that should be moved into the `~/Library/Dictionaries` folder on installation.
| `font`              | yes                           | Relative path to a Font that should be moved into the `~/Library/Fonts` folder on installation.
| `input_method`      | yes                           | Relative path to a Input Method that should be moved into the `~/Library/Input Methods` folder on installation.
| `internet_plugin`   | yes                           | Relative path to a Service that should be moved into the `~/Library/Internet Plug-Ins` folder on installation.
| `manpage`           | yes                           | Relative path to a Man Page that should be linked into the respective man page folder on installation, e.g. `/usr/local/share/man/man3` for `my_app.3`.
| `prefpane`          | yes                           | Relative path to a Preference Pane that should be moved into the `~/Library/PreferencePanes` folder on installation.
| `qlplugin`          | yes                           | Relative path to a QuickLook Plugin that should be moved into the `~/Library/QuickLook` folder on installation.
| `mdimporter`        | yes                           | Relative path to a Spotlight metadata importer that should be moved into the `~/Library/Spotlight` folder on installation.
| `screen_saver`      | yes                           | Relative path to a Screen Saver that should be moved into the `~/Library/Screen Savers` folder on installation.
| `service`           | yes                           | Relative path to a Service that should be moved into the `~/Library/Services` folder on installation.
| `audio_unit_plugin` | yes                           | Relative path to an Audio Unit plugin that should be moved into the `~/Library/Audio/Components` folder on installation.
| `vst_plugin`        | yes                           | Relative path to a VST Plugin that should be moved into the `~/Library/Audio/VST` folder on installation.
| `vst3_plugin`       | yes                           | Relative path to a VST3 Plugin that should be moved into the `~/Library/Audio/VST3` folder on installation.
| `suite`             | yes                           | Relative path to a containing directory that should be moved into the `/Applications` folder on installation.<br />See [Suite Stanza Details](#stanza-suite) for more information.
| `artifact`          | yes                           | Relative path to an arbitrary path that should be moved on installation. Must provide an absolute path as a `target` (example [alcatraz.rb](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/alcatraz.rb#L12)). This is only for unusual cases. The `app` stanza is strongly preferred when moving `.app` bundles.
| `installer`         | yes                           | Describes an executable which must be run to complete the installation.<br />See [Installer Stanza Details](#stanza-installer) for more information.
| `stage_only`        | no                            | `true`. Assert that the Cask contains no activatable artifacts.

### Optional Stanzas

| name                   | multiple occurrences allowed? | value               |
| ---------------------- |------------------------------ | ------------------- |
| `uninstall`            | yes                           | Procedures to uninstall a Cask. Optional unless the `pkg` stanza is used.<br />See [Uninstall Stanza Details](#stanza-uninstall) for more information.
| `zap`                  | yes                           | Additional procedures for a more complete uninstall, including user files and shared resources.<br />See [Zap Stanza Details](#stanza-zap) for more information.
| `appcast`              | no                            | URL providing an appcast feed to find updates for this Cask.<br />See [Appcast Stanza Details](#stanza-appcast) for more information.
| `depends_on`           | yes                           | List of dependencies and requirements for this Cask.<br />See [Depends_on Stanza Details](#stanza-depends_on) for more information.
| `conflicts_with`       | yes                           | List of conflicts with this Cask (*not yet functional*).<br />See [Conflicts_with Stanza Details](#stanza-conflicts_with) for more information.
| `caveats`              | yes                           | String or Ruby block providing the user with Cask-specific information at install time.<br />See [Caveats Stanza Details](#stanza-caveats) for more information.
| `livecheck`            | no                            | Ruby block describing how to find updates for this Cask.<br />See [Livecheck Stanza Details](#stanza-livecheck) for more information.
| `preflight`            | yes                           | Ruby block containing preflight install operations (needed only in very rare cases).
| `postflight`           | yes                           | Ruby block containing postflight install operations.<br />See [Postflight Stanza Details](#stanza-flight) for more information.
| `uninstall_preflight`  | yes                           | Ruby block containing preflight uninstall operations (needed only in very rare cases).
| `uninstall_postflight` | yes                           | Ruby block containing postflight uninstall operations.
| `language`             | required                      | Ruby block, called with language code parameters, containing other stanzas and/or a return value.<br />See [Language Stanza Details](#stanza-language) for more information.
| `container nested:`    | no                            | Relative path to an inner container that must be extracted before moving on with the installation. This allows us to support dmg inside tar, zip inside dmg, etc.
| `container type:`      | no                            | Symbol to override container-type autodetect. May be one of: `:air`, `:bz2`, `:cab`, `:dmg`, `:generic_unar`, `:gzip`, `:otf`, `:pkg`, `:rar`, `:seven_zip`, `:sit`, `:tar`, `:ttf`, `:xar`, `:zip`, `:naked`. (Example: [parse.rb](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/parse.rb#L11))
| `auto_updates`         | no                            | `true`. Assert the Cask artifacts auto-update. Use if `Check for Updates…` or similar is present in app menu, but not if it only opens a webpage and does not do the download and installation for you.


## Stanza descriptions

### Stanza: `app`

In the simple case of a string argument to `app`, the source file is moved to the target `/Applications` directory. For example:

```ruby
app "Alfred 2.app"
```

by default moves the source to:

```bash
/Applications/Alfred 2.app
```

#### Renaming the Target

You can rename the target which appears in your `/Applications` directory by adding a `target:` key to `app`. Example (from [scala-ide.rb](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/scala-ide.rb#L21)):

```ruby
app "eclipse/Eclipse.app", target: "Scala IDE.app"
```

#### target: May Contain an Absolute Path

If `target:` has a leading slash, it is interpreted as an absolute path. The containing directory for the absolute path will be created if it does not already exist. Example (from [manopen.rb](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/manopen.rb#L12)):

```ruby
artifact "openman.1", target: "/usr/local/share/man/man1/openman.1"
```

#### target: Works on Most Artifact Types

The `target:` key works similarly for most Cask artifacts, such as `app`, `binary`, `colorpicker`, `dictionary`, `font`, `input_method`, `prefpane`, `qlplugin`, `mdimporter`, `service`, `suite`, and `artifact`.

#### target: Should Only Be Used in Select Cases

Don’t use `target:` for aesthetic reasons, like removing version numbers (`app "Slack #{version}.app", target: "Slack.app"`). Use it when it makes sense functionally and document your reason clearly in the Cask, using one of the templates: [for clarity](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/imagemin.rb#L12); [for consistency](https://github.com/Homebrew/homebrew-cask/blob/d2a6b26df69fc28c4d84d6f5198b2b652c2f414d/Casks/devonthink-pro-office.rb#L16); [to prevent conflicts](https://github.com/Homebrew/homebrew-cask/blob/bd6dc1a64e0bdd35ba0e20789045ea023b0b6aed/Casks/flash-player-debugger.rb#L11); [due to developer suggestion](https://github.com/Homebrew/homebrew-cask/blob/ff3e9c4a6623af44b8a071027e8dcf3f4edfc6d9/Casks/kivy.rb#L12).

### Stanza: `appcast`

The value of the `appcast` stanza is a string, holding the URL for an appcast which provides information on future updates.

Note: The [`livecheck` stanza](#stanza-livecheck) should be preferred in most cases, as it allows casks to be updated automatically.

The main casks repo only accepts submissions for stable versions of software (and [documented exceptions](https://docs.brew.sh/Acceptable-Casks#but-there-is-no-stable-version)), but it still gets pull requests for unstable versions. By checking the submitted `version` against the contents of an appcast, we can better detect these invalid cases.

Example: [`atom.rb`](https://github.com/Homebrew/homebrew-cask/blob/645dbb8228ec2f1f217ed1431e188687aac13ca5/Casks/atom.rb#L7)

There are a few different ways the `appcast` can be determined:

* If the app is distributed via GitHub releases, the `appcast` will be of the form `https://github.com/<user>/<project_name>/releases.atom`. Example: [`electron.rb`](https://github.com/Homebrew/homebrew-cask/blob/645dbb8228ec2f1f217ed1431e188687aac13ca5/Casks/electron.rb#L7)

* If the app is distributed via GitLab releases, the `appcast` will be of the form `https://gitlab.com/<user>/<project_name>/-/tags?format=atom`. Example: [`grafx.rb`](https://github.com/Homebrew/homebrew-cask/blob/b22381902f9da870bb07d21b496558f283dad612/Casks/grafx.rb#L6)

* The popular update framework [Sparkle](https://sparkle-project.org/) generally uses the `SUFeedURL` property in `Contents/Info.plist` inside `.app` bundles. Example: [`glyphs.rb`](https://github.com/Homebrew/homebrew-cask/blob/645dbb8228ec2f1f217ed1431e188687aac13ca5/Casks/glyphs.rb#L6)

* Sourceforge projects follow the form `https://sourceforge.net/projects/<project_name>/rss`. A more specific page can be used as needed, pointing to a specific directory structure: `https://sourceforge.net/projects/<project_name>/rss?path=/<path_here>`. Example: [`seashore.rb`](https://github.com/Homebrew/homebrew-cask/blob/645dbb8228ec2f1f217ed1431e188687aac13ca5/Casks/seashore.rb#L6)

* An appcast can be any URL hosted by the app’s developer that changes every time a new release is out or that contains the version number of the current release (e.g. a download HTML page). Webpages that only change on new version releases are preferred, as are sites that do not contain previous version strings (i.e. avoid changelog pages if the download page contains the current version number but not older ones). Example: [`razorsql.rb`](https://github.com/Homebrew/homebrew-cask/blob/645dbb8228ec2f1f217ed1431e188687aac13ca5/Casks/razorsql.rb#L6)

The [`find-appcast`](https://github.com/Homebrew/homebrew-cask/blob/HEAD/developer/bin/find-appcast) script is able to identify some of these, as well as `electron-builder` appcasts which are trickier to find by hand. Run it with `"$(brew --repository homebrew/cask)/developer/bin/find-appcast" '</path/to/software.app>'`.

#### Parameters

| key             | value       |
| --------------- | ----------- |
| `must_contain:` | a custom string for `brew audit --appcast <cask>` to check against. |

Sometimes a `version` doesn’t match a string on the webpage, in which case we tweak what to search for. Example: if `version` is `6.26.1440` and the appcast’s contents only show `6.24`, the check for “is `version` in the appcast feed” will fail. With `must_contain`, the check is told to “look for this string instead of `version`”. In the example, `must_contain: version.major_minor` is saying “look for `6.24`”, making the check succeed.

If no `must_contain` is given, the check considers from the beginning of the `version` string until the first character that isn’t alphanumeric or a period. Example: if `version` is `6.26b-14,40`, the check will see `6.26b`. This is so it covers most cases by default, while still allowing complex `version`s suitable for interpolation on the rest of the cask.

Example of using `must_contain`: [`hwsensors.rb`](https://github.com/Homebrew/homebrew-cask/blob/87bc3860f43d5b14d0c38ae8de469d24ee7f5b2f/Casks/hwsensors.rb#L6L7)

### Stanza: `binary`

In the simple case of a string argument to `binary`, the source file is linked into the `$(brew --prefix)/bin` directory (typically `/usr/local/bin`) on installation. For example (from [operadriver.rb](https://github.com/Homebrew/homebrew-cask/blob/60531a2812005dd5f17dc92f3ce7419af3c5d019/Casks/operadriver.rb#L11)):

```ruby
binary "operadriver"
```

creates a symlink to:

```bash
$(brew --prefix)/bin/operadriver
```

from a source file such as:

```bash
/usr/local/Caskroom/operadriver/0.2.2/operadriver
```

A binary (or multiple) can also be contained in an application bundle:

```ruby
app "Atom.app"
binary "#{appdir}/Atom.app/Contents/Resources/app/apm/bin/apm"
```

You can rename the target which appears in your binaries directory by adding a `target:` key to `binary`:

```ruby
binary "#{appdir}/Atom.app/Contents/Resources/app/atom.sh", target: "atom"
```

Behaviour and usage of `target:` is [the same as with `app`](#renaming-the-target). However, for `binary` the select cases don’t apply as rigidly. It’s fine to take extra liberties with `target:` to be consistent with other command-line tools, like [changing case](https://github.com/Homebrew/homebrew-cask/blob/9ad93b833961f1d969505bc6bdb1c2ad4e58a433/Casks/openscad.rb#L12), [removing an extension](https://github.com/Homebrew/homebrew-cask/blob/c443d4f5c6864538efe5bb1ecf662565a5ffb438/Casks/filebot.rb#L13), or [cleaning up the name](https://github.com/Homebrew/homebrew-cask/blob/146917cbcc679648de6b0bccff4e9b43fce0e6c8/Casks/minishift.rb#L13).

### Stanza: `caveats`

Sometimes there are particularities with the installation of a piece of software that cannot or should not be handled programmatically by Homebrew Cask. In those instances, `caveats` is the way to inform the user. Information in `caveats` is displayed when a cask is invoked with either `install` or `info`.

To avoid flooding users with too many messages (thus desensitising them to the important ones), `caveats` should be used sparingly and exclusively for installation-related matters. If you’re not sure a `caveat` you find pertinent is installation-related or not, ask a maintainer. As a general rule, if your case isn’t already covered in our comprehensive [`caveats Mini-DSL`](#caveats-mini-dsl), it’s unlikely to be accepted.

#### caveats as a String

When `caveats` is a string, it is evaluated at compile time. The following methods are available for interpolation if `caveats` is placed in its customary position at the end of the Cask:

| method             | description |
| ------------------ | ----------- |
| `token`            | the Cask token
| `version`          | the Cask version
| `homepage`         | the Cask homepage
| `caskroom_path`    | the containing directory for this Cask, typically `/usr/local/Caskroom/<token>` (only available with block form)
| `staged_path`      | the staged location for this Cask, including version number: `/usr/local/Caskroom/<token>/<version>` (only available with block form)

Example:

```ruby
caveats "Using #{token} is hazardous to your health."
```

#### caveats as a Block

When `caveats` is a Ruby block, evaluation is deferred until install time. Within a block you may refer to the `@cask` instance variable, and invoke any method available on `@cask`.

#### caveats Mini-DSL

There is a mini-DSL available within `caveats` blocks.

The following methods may be called to generate standard warning messages:

| method                             | description |
| ---------------------------------- | ----------- |
| `path_environment_variable "path"` | users should make sure `path` is in their `$PATH` environment variable.
| `zsh_path_helper "path"`           | zsh users must take additional steps to make sure `path` is in their `$PATH` environment variable.
| `depends_on_java "version"`        | users should make sure they have the specified version of java installed. `version` can be exact (e.g. `6`), a minimum (e.g. `7+`), or omitted (when any version works).
| `logout`                           | users should log out and log back in to complete installation.
| `reboot`                           | users should reboot to complete installation.
| `files_in_usr_local`               | the Cask installs files to `/usr/local`, which may confuse Homebrew.
| `discontinued`                     | all software development has been officially discontinued upstream.
| `free_license "web_page"`          | users may get an official license to use the software at `web_page`.
| `kext`                             | users may need to enable their kexts in System Preferences → Security & Privacy → General.
| `unsigned_accessibility`           | users will need to re-enable the app on each update in System Preferences → Security & Privacy → Privacy as it is unsigned.
| `license "web_page"`               | software has a usage license at `web_page`.

Example:

```ruby
caveats do
  path_environment_variable "/usr/texbin"
end
```

### Stanza: `conflicts_with`

`conflicts_with` is used to declare conflicts that keep a Cask from installing or working correctly.

#### conflicts_with cask:

The value should be another Cask token.

Example use: [`wireshark`](https://github.com/Homebrew/homebrew-cask/blob/903493e09cf33b845e7cf497ecf9cfc9709087ee/Casks/wireshark.rb#L10), which conflicts with `wireshark-chmodbpf`.

```ruby
conflicts_with cask: "wireshark-chmodbpf"
```

#### conflicts_with formula:

Note: `conflicts_with formula:` is a stub and is not yet functional.

The value should be another formula name.

Example use: [`macvim`](https://github.com/Homebrew/homebrew-cask/blob/84b90afd7b571e581f8a48d4bdf9c7bb24ebff3b/Casks/macvim.rb#L10), which conflicts with the `macvim` formula.

```ruby
conflicts_with formula: "macvim"
```

### Stanza: `depends_on`

`depends_on` is used to declare dependencies and requirements for a Cask.
`depends_on` is not consulted until `install` is attempted.

#### depends_on cask:

The value should be another Cask token, needed by the current Cask.

Example use: [`cellery`](https://github.com/Homebrew/homebrew-cask/blob/4002df8f6bca93ed6eb40494995fcfa038cf99bf/Casks/cellery.rb#L11) depends on OSXFUSE:

```ruby
depends_on cask: "osxfuse"
```

#### depends_on formula:

The value should name a Homebrew Formula needed by the Cask.

Example use: some distributions are contained in archive formats such as `7z` which are not supported by stock Apple tools. For these cases, a more capable archive reader may be pulled in at install time by declaring a dependency on the Homebrew Formula `unar`:

```ruby
depends_on formula: "unar"
```

#### depends_on macos:

##### Requiring an Exact macOS Release

The value for `depends_on macos:` may be a symbol or an array of symbols, listing the exact compatible macOS releases.

The available values for macOS releases are:

| symbol             | corresponding release
| -------------------|----------------------
| `:yosemite`        | `10.10`
| `:el_capitan`      | `10.11`
| `:sierra`          | `10.12`
| `:high_sierra`     | `10.13`
| `:mojave`          | `10.14`
| `:catalina`        | `10.15`
| `:big_sur`         | `11.0`
| `:monterey`        | `12.0`

Only major releases are covered (version numbers containing a single dot). The symbol form is used for readability. The following are all valid ways to enumerate the exact macOS release requirements for a Cask:

```ruby
depends_on macos: :big_sur
depends_on macos: [
  :catalina,
  :big_sur,
]
```

##### Setting a Minimum macOS Release

`depends_on macos:` can also accept a string starting with a comparison operator such as `>=`, followed by an macOS release in the form above. The following is a valid expression meaning “at least macOS Big Sur (11.0)”:

```ruby
depends_on macos: ">= :big_sur"
```

A comparison expression cannot be combined with any other form of `depends_on macos:`.

#### depends_on arch:

The value for `depends_on arch:` may be a symbol or an array of symbols, listing the hardware compatibility requirements for a Cask. The requirement is satisfied at install time if any one of multiple `arch:` value matches the user’s hardware.

The available symbols for hardware are:

| symbol     | meaning        |
| ---------- | -------------- |
| `:x86_64`  | 64-bit Intel   |
| `:intel`   | 64-bit Intel   |

The following are all valid expressions:

```ruby
depends_on arch: :intel
depends_on arch: :x86_64            # same meaning as above
depends_on arch: [:x86_64]          # same meaning as above
```

Since as of now all the macOS versions we support only run on 64-bit Intel, `depends_on arch:` is never necessary.

#### All depends_on Keys

| key        | description |
| ---------- | ----------- |
| `formula:` | a Homebrew Formula
| `cask:`    | a Cask token
| `macos:`   | a symbol, string, array, or comparison expression defining macOS release requirements
| `arch:`    | a symbol or array defining hardware requirements
| `java:`    | *stub - not yet functional*

### Stanza: `desc`

`desc` accepts a single-line UTF-8 string containing a short description of the software. It’s used to help with searchability and disambiguation, thus it must concisely describe what the software does (or what you can accomplish with it).

`desc` is not for app slogans! Vendors’ descriptions tend to be filled with generic adjectives such as “modern” and “lightweight”. Those are meaningless marketing fluff (do you ever see apps proudly describing themselves as outdated and bulky?) which must the deleted. It’s fine to use the information on the software’s website as a starting point, but it will require editing in almost all cases.

#### Dos and Don'ts

- **Do** start with an uppercase letter.

  ```diff
  - desc "sound and music editor"
  + desc "Sound and music editor"
  ```

- **Do** be brief, i.e. use less than 80 characters.

  ```diff
  - desc "Sound and music editor which comes with effects, instruments, sounds and all kinds of creative features"
  + desc "Sound and music editor"
  ```

- **Do** describe what the software does or is:

  ```diff
  - desc "Development of musical ideas made easy"
  + desc "Sound and music editor"
  ```

- **Do not** include the platform. Casks only work on macOS, so this is redundant information.

  ```diff
  - desc "Sound and music editor for macOS"
  + desc "Sound and music editor"
  ```

- **Do not** include the Cask’s [name](#stanza-name).

  ```diff
  - desc "Ableton Live is a sound and music editor"
  + desc "Sound and music editor"
  ```

- **Do not** include the vendor. This should be added to the Cask’s [name](#stanza-name) instead.


  ```diff
  - desc "Sound and music editor made by Ableton"
  + desc "Sound and music editor"
  ```

- **Do not** add user pronouns.

  ```diff
  - desc "Edit your music files"
  + desc "Sound and music editor"
  ```

- **Do not** use empty marketing jargon.

  ```diff
  - desc "Beautiful and powerful modern sound and music editor"
  + desc "Sound and music editor"
  ```

### Stanza: `\*flight`

#### Evaluation of Blocks is Always Deferred

The Ruby blocks defined by `preflight`, `postflight`, `uninstall_preflight`, and `uninstall_postflight` are not evaluated until install time or uninstall time. Within a block, you may refer to the `@cask` instance variable, and invoke any method available on `@cask`.

#### \*flight Mini-DSL

There is a mini-DSL available within these blocks.

The following methods may be called to perform standard tasks:

| method                                    | availability                                     | description |
| ----------------------------------------- | ------------------------------------------------ | ----------- |
| `set_ownership(paths)`                    | `preflight`, `postflight`, `uninstall_preflight` | set user and group ownership of `paths`. Example: [`unifi-controller.rb`](https://github.com/Homebrew/homebrew-cask/blob/8a452a41707af6a661049da6254571090fac5418/Casks/unifi-controller.rb#L13)
| `set_permissions(paths, permissions_str)` | `preflight`, `postflight`, `uninstall_preflight` | set permissions in `paths` to `permissions_str`. Example: [`docker-machine.rb`](https://github.com/Homebrew/homebrew-cask/blob/8a452a41707af6a661049da6254571090fac5418/Casks/docker-machine.rb#L16)

`set_ownership(paths)` defaults user ownership to the current user and group ownership to `staff`. These can be changed by passing in extra options: `set_ownership(paths, user: 'user', group: 'group')`.

### Stanza: `installer`

This stanza must always be accompanied by [`uninstall`](#stanza-uninstall).

The `installer` stanza takes a series of key-value pairs, the first key of which must be `manual:` or `script:`.

#### installer manual:

`installer manual:` takes a single string value, describing a GUI installer which must be run by the user at a later time. The path may be absolute, or relative to the Cask. Example (from [nutstore.rb](https://github.com/Homebrew/homebrew-cask/blob/249ec31048591308e63e50f79dae01d2f933cccf/Casks/nutstore.rb#L9)):

```ruby
installer manual: "Nutstore Installer.app"
```

#### installer script:

`installer script:` introduces a series of key-value pairs describing a command which will automate completion of the install. **It should never be used for interactive installations.** The form is similar to `uninstall script:`:

| key             | value
| ----------------|------------------------------
| `executable:`   | path to an install script to be run
| `args:`         | array of arguments to the install script
| `input:`        | array of lines of input to be sent to `stdin` of the script
| `must_succeed:` | set to `false` if the script is allowed to fail
| `sudo:`         | set to `true` if the script needs `sudo`

The path may be absolute, or relative to the Cask. Example (from [miniforge.rb](https://github.com/Homebrew/homebrew-cask/blob/ed2033fb3578376c3ee58a2cb459ef96fa6eb37d/Casks/miniforge.rb#L15L18)):

```ruby
  installer script: {
    executable: "Miniforge3-#{version}-MacOSX-x86_64.sh",
    args:       ["-b", "-p", "#{caskroom_path}/base"],
  }
```

If the `installer script:` does not require any of the key-values it can point directly to the path of the install script:

```ruby
installer script: "#{staged_path}/install.sh"
```

### Stanza: `language`

The `language` stanza can match [ISO 639-1](https://en.wikipedia.org/wiki/ISO_639-1) language codes, regional identifiers ([ISO 3166-1 Alpha 2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)) and script codes ([ISO 15924](https://en.wikipedia.org/wiki/ISO_15924)), or a combination thereof.

US English should always be used as the default language:

```ruby
language "zh", "CN" do
  "zh_CN"
end

language "de" do
  "de_DE"
end

language "en-GB" do
  "en_GB"
end

language "en", default: true do
  "en_US"
end
```

Note that the following are not the same:

```ruby
language "en", "GB" do
  # matches all locales containing "en" or "GB"
end

language "en-GB" do
  # matches only locales containing "en" and "GB"
end
```

The return value of the matching `language` block can be accessed by simply calling `language`.

```ruby
homepage "https://example.org/#{language}"
```

Examples: [Firefox](https://github.com/Homebrew/homebrew-cask/blob/306b8fbd9502036f1ca742f70c569d8677b62403/Casks/firefox.rb#L4L74), [Battle.net](https://github.com/Homebrew/homebrew-cask/blob/306b8fbd9502036f1ca742f70c569d8677b62403/Casks/battle-net.rb#L5L17)


#### Installation

To install a cask in a specific language, you can pass the `--language=` option to `brew install`:

```
brew install firefox --language=it
```

### Stanza: `livecheck`

The `livecheck` stanza is used to automatically fetch the latest version of a cask from changelogs, release notes, appcasts, etc. See also: [`brew livecheck` reference](Brew-Livecheck.md)

Every `livecheck` block must contain a `url`, which can either be a string or a symbol pointing to other URLs in the cask (`:url` or `:homepage`).

Additionally, a `livecheck` should specify which `strategy` should be used to extract the version:

| `strategy`      | Description |
|-----------------|-----------|
| `:header_match` | extract version from HTTP headers (e.g. `Location` or `Content-Disposition`) |
| `:page_match`   | extract version from page contents                                           |
| `:sparkle`      | extract version from Sparkle appcast contents                                |

Here is a basic example, extracting a simple version from a page:

```ruby
livecheck do
  url "https://example.org/my-app/download"
  strategy :page_match
  regex(%r{href=.*?/MyApp-(\d+(?:\.\d+)*)\.zip}i)
end
```

If the download URL is present on the homepage, we can use a symbol instead of a string:

```ruby
livecheck do
  url :homepage
  strategy :page_match
  regex(%r{href=.*?/MyApp-(\d+(?:\.\d+)*)\.zip}i)
end
```


The `header_match` strategy will try parsing a version from the filename (in the `Content-Disposition` header) and the final URL (in the `Location` header). If that doesn't work, a `regex` can be specified, e.g.:

```ruby
strategy :header_match
regex(/MyApp-(\d+(?:\.\d+)*)\.zip/i)
```

If the version depends on multiple header fields, a block can be specified, e.g.

```ruby
strategy :header_match do |headers|
  v = headers["content-disposition"][/MyApp-(\d+(?:\.\d+)*)\.zip/i, 1]
  id = headers["location"][%r{/(\d+)/download$}i, 1]
  next if v.blank? || id.blank?
  
  "#{v},#{id}"
end
```

Similarly, the `:page_match` strategy can also be used for more complex versions by specifying a block:

```ruby
strategy :page_match do |page|
  match = page.match(%r{href=.*?/(\d+)/MyApp-(\d+(?:\.\d+)*)\.zip}i)
  next if match.blank?
  
  "#{match[2]},#{match[1]}"
end
```

### Stanza: `name`

`name` accepts a UTF-8 string defining the name of the software, including capitalization and punctuation. It is used to help with searchability and disambiguation.

Unlike the [token](#token-reference), which is simplified and reduced to a limited set of characters, the `name` stanza can include the proper capitalization, spacing and punctuation to match the official name of the software. For disambiguation purposes, it is recommended to spell out the name of the application, and including the vendor name if necessary. A good example is [`pycharm-ce`](https://github.com/Homebrew/homebrew-cask/blob/fc05c0353aebb28e40db72faba04b82ca832d11a/Casks/pycharm-ce.rb#L6-L7), whose name is spelled out as `Jetbrains PyCharm Community Edition`, even though it is likely never referenced as such anywhere.

Additional details about the software can be provided in the [desc](#stanza-desc) stanza.

The `name` stanza can be repeated multiple times if there are useful alternative names. The first instance should use the Latin alphabet. For example, see the [`cave-story`](https://github.com/Homebrew/homebrew-cask/blob/0fe48607f5656e4f1de58c6884945378b7e6f960/Casks/cave-story.rb#L7-L9) cask, whose original name does not use the Latin alphabet.

### Stanza: `pkg`

This stanza must always be accompanied by [`uninstall`](#stanza-uninstall)

The first argument to the `pkg` stanza should be a relative path to the `.pkg` file to be installed. For example:

```ruby
pkg "Unity.pkg"
```

Subsequent arguments to `pkg` are key/value pairs which modify the install process. Currently supported keys are `allow_untrusted:` and `choices:`.

#### `pkg allow_untrusted:`

`pkg allow_untrusted: true` can be used to install the `.pkg` with an untrusted certificate passing `-allowUntrusted` to `/usr/sbin/installer`.

This option is not permitted in official Homebrew Cask taps, it is only provided for use in third-party taps or local Casks.

Example ([alinof-timer.rb](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/alinof-timer.rb#L10)):

```ruby
pkg "AlinofTimer.pkg", allow_untrusted: true
```

#### `pkg choices:`

`pkg choices:` can be used to override `.pkg`’s default install options via `-applyChoiceChangesXML`. It uses a deserialized version of the `choiceChanges` property list (refer to the `CHOICE CHANGES FILE` section of the `installer` manual page by running `man -P 'less --pattern "^CHOICE CHANGES FILE"' installer`).

Running the  macOS command:

```bash
$ installer -showChoicesXML -pkg '/path/to/my.pkg'
```

will output an XML which you can use to extract the `choices:` values, as well as their equivalents to the GUI options.

See [this pull request for wireshark-chmodbpf](https://github.com/Homebrew/homebrew-cask/pull/26997) and [this one for wine-staging](https://github.com/Homebrew/homebrew-cask/pull/27937) for some examples of the procedure.

Example ([wireshark-chmodbpf.rb](https://github.com/Homebrew/homebrew-cask/blob/f95b8a8306b91fe9da7908b842f4a5fa80f7afe0/Casks/wireshark-chmodbpf.rb#L9-L26)):
```ruby
pkg "Wireshark #{version} Intel 64.pkg",
    choices: [
               {
                 "choiceIdentifier" => "wireshark",
                 "choiceAttribute"  => "selected",
                 "attributeSetting" => 0,
               },
               {
                 "choiceIdentifier" => "chmodbpf",
                 "choiceAttribute"  => "selected",
                 "attributeSetting" => 1,
               },
               {
                 "choiceIdentifier" => "cli",
                 "choiceAttribute"  => "selected",
                 "attributeSetting" => 0,
               },
             ]
```

Example ([wine-staging.rb](https://github.com/Homebrew/homebrew-cask/blob/51b65f6a5a25a7f79af4d372e1a0bf1dc3849251/Casks/wine-staging.rb#L11-L18)):
```ruby
pkg "winehq-staging-#{version}.pkg",
    choices: [
               {
                 "choiceIdentifier" => "choice3",
                 "choiceAttribute"  => "selected",
                 "attributeSetting" => 1,
               },
             ]
```

### Stanza: `sha256`

#### Calculating the SHA256

The `sha256` value is usually calculated by the command:

```bash
$ shasum --algorithm 256 <file>
```

#### Special Value `:no_check`

The special value `sha256 :no_check` is used to turn off SHA checking whenever checksumming is impractical due to the upstream configuration.

`version :latest` requires `sha256 :no_check`, and this pairing is common. However, `sha256 :no_check` does not require `version :latest`.

We use a checksum whenever possible.

### Stanza: `suite`

Some distributions provide a suite of multiple applications, or an application with required data, to be installed together in a subdirectory of `/Applications`.

For these Casks, use the `suite` stanza to define the directory containing the application suite. Example (from [sketchup.rb](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/sketchup.rb#L12)):

```ruby
suite "SketchUp 2016"
```

The value of `suite` is never an `.app` bundle, but a plain directory.

### Stanza: `uninstall`

> If you cannot design a working `uninstall` stanza, please submit your cask anyway. The maintainers can help you write an `uninstall` stanza, just ask!

#### `uninstall pkgutil:` Is The Easiest and Most Useful

`pkgutil:` is the easiest and most useful `uninstall` directive. See [Uninstall Key pkgutil:](#uninstall-key-pkgutil).

#### `uninstall` Is Required for Casks That Install a pkg or installer manual:

For most Casks, uninstall actions are determined automatically, and an explicit `uninstall` stanza is not needed. However, a Cask which uses the `pkg` or `installer manual:` stanzas will **not** know how to uninstall correctly unless an `uninstall` stanza is given.

So, while the [Cask DSL](#required-stanzas) does not enforce the requirement, it is much better for end-users if every `pkg` and `installer manual:` has a corresponding `uninstall`.

The `uninstall` stanza is available for non-`pkg` Casks, and is useful for a few corner cases. However, the documentation below concerns the typical case of using `uninstall` to define procedures for a `pkg`.

#### There Are Multiple Uninstall Techniques

Since `pkg` installers can do arbitrary things, different techniques are needed to uninstall in each case. You may need to specify one, or several, of the following key/value pairs as arguments to `uninstall`.

#### Summary of Keys

* `early_script:` (string or hash) - like [`script:`](#uninstall-key-script), but runs early (for special cases, best avoided)
* [`launchctl:`](#uninstall-key-launchctl) (string or array) - ids of `launchctl` jobs to remove
* [`quit:`](#uninstall-key-quit) (string or array) - bundle ids of running applications to quit
* [`signal:`](#uninstall-key-signal) (array of arrays) - signal numbers and bundle ids of running applications to send a Unix signal to (used when `quit:` does not work)
* [`login_item:`](#uninstall-key-login_item) (string or array) - names of login items to remove
* [`kext:`](#uninstall-key-kext) (string or array) - bundle ids of kexts to unload from the system
* [`script:`](#uninstall-key-script) (string or hash) - relative path to an uninstall script to be run via sudo; use hash if args are needed
  - `executable:` - relative path to an uninstall script to be run via sudo (required for hash form)
  - `args:` - array of arguments to the uninstall script
  - `input:` - array of lines of input to be sent to `stdin` of the script
  - `must_succeed:` - set to `false` if the script is allowed to fail
  - `sudo:` - set to `true` if the script needs `sudo`
* [`pkgutil:`](#uninstall-key-pkgutil) (string, regexp or array of strings and regexps) - strings or regexps matching bundle ids of packages to uninstall using `pkgutil`
* [`delete:`](#uninstall-key-delete) (string or array) - single-quoted, absolute paths of files or directory trees to remove. `delete:` should only be used as a last resort. `pkgutil:` is strongly preferred.
* `rmdir:` (string or array) - single-quoted, absolute paths of directories to remove if empty. Works recursively.
* [`trash:`](#uninstall-key-trash) (string or array) - single-quoted, absolute paths of files or directory trees to move to Trash.

Each `uninstall` technique is applied according to the order above. The order in which `uninstall` keys appear in the Cask file is ignored.

For assistance filling in the right values for `uninstall` keys, there are several helper scripts found under `developer/bin` in the Homebrew Cask repository. Each of these scripts responds to the `-help` option with additional documentation.

The easiest way to work out an `uninstall` stanza is on a system where the `pkg` is currently installed and operational. To operate on an uninstalled `pkg` file, see [Working With a pkg File Manually](#working-with-a-pkg-file-manually), below.

#### `uninstall` Key `pkgutil:`

This is the most useful uninstall key. `pkgutil:` is often sufficient to completely uninstall a `pkg`, and is strongly preferred over `delete:`.

IDs for the most recently-installed packages can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_recent_pkg_ids"
```

`pkgutil:` also accepts a regular expression match against multiple package IDs. The regular expressions are somewhat nonstandard. To test a `pkgutil:` regular expression against currently-installed packages, use the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_pkg_ids_by_regexp" <regular-expression>
```

#### List Files Associated With a pkg Id

Once you know the ID for an installed package, (above), you can list all files on your system associated with that package ID using the macOS command:

```bash
$ pkgutil --files <package.id.goes.here>
```

Listing the associated files can help you assess whether the package included any `launchctl` jobs or kernel extensions (kexts).

#### `uninstall` Key `launchctl:`

IDs for currently loaded `launchctl` jobs can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_loaded_launchjob_ids"
```

IDs for all installed `launchctl` jobs can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_installed_launchjob_ids"
```

#### `uninstall` Key `quit:`

Bundle IDs for currently running Applications can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_running_app_ids"
```

Bundle IDs inside an Application bundle on disk can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_ids_in_app" '/path/to/application.app'
```

#### `uninstall` Key `signal:`

`signal:` should only be needed in the rare case that a process does not respond to `quit:`.

Bundle IDs for `signal:` targets may be obtained as for `quit:`. The value for `signal:` is an array-of-arrays, with each cell containing two elements: the desired Unix signal followed by the corresponding bundle ID.

The Unix signal may be given in numeric or string form (see the `kill` man page for more details).

The elements of the `signal:` array are applied in order, only if there is an existing process associated the bundle ID, and stopping when that process terminates. A bundle ID may be repeated to send more than one signal to the same process.

It is better to use the least-severe signals which are sufficient to stop a process. The `KILL` signal in particular can have unwanted side-effects.

An example, with commonly-used signals in ascending order of severity:

```ruby
  uninstall signal: [
                      ["TERM", "fr.madrau.switchresx.daemon"],
                      ["QUIT", "fr.madrau.switchresx.daemon"],
                      ["INT",  "fr.madrau.switchresx.daemon"],
                      ["HUP",  "fr.madrau.switchresx.daemon"],
                      ["KILL", "fr.madrau.switchresx.daemon"],
                    ]
```

Note that when multiple running processes match the given Bundle ID, all matching processes will be signaled.

Unlike `quit:` directives, Unix signals originate from the current user, not from the superuser. This is construed as a safety feature, since the superuser is capable of bringing down the system via signals. However, this inconsistency may also be considered a bug, and should be addressed in some fashion in a future version.

## `uninstall` key `login_item:`

Login items associated with an Application bundle on disk can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_login_items_for_app" '/path/to/application.app'
```

Note that you will likely need to have opened the app at least once for any login items to be present.

#### `uninstall` Key `kext:`

IDs for currently loaded kernel extensions can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_loaded_kext_ids"
```

IDs inside a kext bundle you have located on disk can be listed using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_id_in_kext" '/path/to/name.kext'
```

#### `uninstall` Key `script:`

`uninstall script:` introduces a series of key-value pairs describing a command which will automate completion of the uninstall. Example (from [gpgtools.rb](https://github.com/Homebrew/homebrew-cask/blob/4a0a49d1210a8202cbdd54bce2986f15049b8b61/Casks/gpgtools.rb#L33-#L37)):

```ruby
  uninstall script:    {
                         executable: "#{staged_path}/Uninstall.app/Contents/Resources/GPG Suite Uninstaller.app/Contents/Resources/uninstall.sh",
                         sudo:       true,
                       }
```

It is important to note that, although `script:` in the above example does attempt to completely uninstall the `pkg`, it should not be used in detriment of [`pkgutil:`](#uninstall-key-pkgutil), but as a complement when possible.

#### `uninstall` Key `delete:`

`delete:` should only be used as a last resort, if other `uninstall` methods are insufficient.

Arguments to `uninstall delete:` should use the following basic rules:

* Basic tilde expansion is performed on paths, i.e. leading `~` is expanded to the home directory.
* Paths must be absolute.
* Glob expansion is performed using the [standard set of characters](https://en.wikipedia.org/wiki/Glob_(programming)).

To remove user-specific files, use the [`zap` stanza](#stanza-zap).

#### `uninstall` Key `trash:`

`trash:` arguments follow the same rules listed above for `delete:`.

#### Working With a pkg File Manually

Advanced users may wish to work with a `pkg` file manually, without having the package installed.

A list of files which may be installed from a `pkg` can be extracted using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_payload_in_pkg" '/path/to/my.pkg'
```

Candidate application names helpful for determining the name of a Cask may be extracted from a `pkg` file using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_apps_in_pkg" '/path/to/my.pkg'
```

Candidate package IDs which may be useful in a `pkgutil:` key may be extracted from a `pkg` file using the command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/list_ids_in_pkg" '/path/to/my.pkg'
```

A fully manual method for finding bundle ids in a package file follows:

1. Unpack `/path/to/my.pkg` (replace with your package name) with `pkgutil --expand /path/to/my.pkg /tmp/expanded.unpkg`.
2. The unpacked package is a folder. Bundle ids are contained within files named `PackageInfo`. These files can be found with the command `find /tmp/expanded.unpkg -name PackageInfo`.
3. `PackageInfo` files are XML files, and bundle ids are found within the `identifier` attributes of `<pkg-info>` tags that look like `<pkg-info ... identifier="com.oracle.jdk7u51" ... >`, where extraneous attributes have been snipped out and replaced with ellipses.
4. Kexts inside packages are also described in `PackageInfo` files. If any kernel extensions are present, the command `find /tmp/expanded.unpkg -name PackageInfo -print0 | xargs -0 grep -i kext` should return a `<bundle id>` tag with a `path` attribute that contains a `.kext` extension, for example `<bundle id="com.wavtap.driver.WavTap" ... path="./WavTap.kext" ... />`.
5. Once bundle ids have been identified, the unpacked package directory can be deleted.

### Stanza: `url`

#### HTTPS URLs are Preferred

If available, an HTTPS URL is preferred. A plain HTTP URL should only be used in the absence of a secure alternative.

#### Additional HTTP/S URL Parameters

When a plain URL string is insufficient to fetch a file, additional information may be provided to the `curl`-based downloader, in the form of key/value pairs appended to `url`:

| key                | value       |
| ------------------ | ----------- |
| `verified:`        | a string repeating the beginning of `url`, for verification purposes. [See below](#when-url-and-homepage-domains-differ-add-verified).
| `using:`           | the symbol `:post` is the only legal value
| `cookies:`         | a hash of cookies to be set in the download request
| `referer:`         | a string holding the URL to set as referer in the download request
| `header:`          | a string holding the header to set for the download request.
| `user_agent:`      | a string holding the user agent to set for the download request. Can also be set to the symbol `:fake`, which will use a generic Browser-like user agent string. We prefer `:fake` when the server does not require a specific user agent.
| `data:`            | a hash of parameters to be set in the POST request

Example of using `cookies:`: [java.rb](https://github.com/Homebrew/homebrew-cask/blob/472930df191d66747a57d5c96c0d00511d56e21b/Casks/java.rb#L5-L8)

Example of using `referer:`: [rrootage.rb](https://github.com/Homebrew/homebrew-cask/blob/312ae841f1f1b2ec07f4d88b7dfdd7fbdf8d4f94/Casks/rrootage.rb#L5)

Example of using `header:`: [issue-325182724](https://github.com/Homebrew/brew/pull/6545#issue-325182724)

#### When URL and Homepage Domains Differ, Add `verified:`

When the domains of `url` and `homepage` differ, the discrepancy should be documented with the `verified:` parameter, repeating the smallest possible portion of the URL that uniquely identifies the app or vendor, excluding the protocol. Example: [`shotcut.rb`](https://github.com/Homebrew/homebrew-cask/blob/08733296b49c59c58b6beeada59ed4207cef60c3/Casks/shotcut.rb#L5L6).

This must be added so a user auditing the cask knows the URL was verified by the Homebrew Cask team as the one provided by the vendor, even though it may look unofficial. It is our responsibility as Homebrew Cask maintainers to verify both the `url` and `homepage` information when first added (or subsequently modified, apart from versioning).

The parameter doesn’t mean you should trust the source blindly, but we only approve casks in which users can easily verify its authenticity with basic means, such as checking the official homepage or public repository. Occasionally, slightly more elaborate techniques may be used, such as inspecting an [`appcast`](#stanza-appcast) we established as official. Cases where such quick verifications aren’t possible (e.g. when the download URL is behind a registration wall) are [treated in a stricter manner](https://docs.brew.sh/Acceptable-Casks#unofficial-vendorless-and-walled-builds).

#### Difficulty Finding a URL

Web browsers may obscure the direct `url` download location for a variety of reasons. Homebrew Cask supplies a script which can read extended file attributes to extract the actual source URL for most files downloaded by a browser on macOS. The script usually emits multiple candidate URLs; you may have to test each of them:

```bash
$ $(brew --repository homebrew/cask)/developer/bin/list_url_attributes_on_file <file>
```

#### Subversion URLs

In rare cases, a distribution may not be available over ordinary HTTP/S. Subversion URLs are also supported, and can be specified by appending the following key/value pairs to `url`:

| key                | value       |
| ------------------ | ----------- |
| `using:`           | the symbol `:svn` is the only legal value
| `revision:`        | a string identifying the subversion revision to download
| `trust_cert:`      | set to `true` to automatically trust the certificate presented by the server (avoiding an interactive prompt)

#### SourceForge/OSDN URLs

SourceForge and OSDN (formerly `SourceForge.JP`) projects are common ways to distribute binaries, but they provide many different styles of URLs to get to the goods.

We prefer URLs of this format:

```
https://downloads.sourceforge.net/<project_name>/<filename>.<ext>
```

Or, if it’s from [OSDN](https://osdn.jp/):

```
http://<subdomain>.osdn.jp/<project_name>/<release_id>/<filename>.<ext>
```

`<subdomain>` is typically of the form `dl` or `<user>.dl`.

If these formats are not available, and the application is macOS-exclusive (otherwise a command-line download defaults to the Windows version) we prefer the use of this format:

```
https://sourceforge.net/projects/<project_name>/files/latest/download
```

#### Some Providers Block Command-line Downloads

Some hosting providers actively block command-line HTTP clients. Such URLs cannot be used in Casks.

Other providers may use URLs that change periodically, or even on each visit (example: FossHub). While some cases [could be circumvented](#using-a-block-to-defer-code-execution), they tend to occur when the vendor is actively trying to prevent automated downloads, so we prefer to not add those casks to the main repository.

#### Using a Block to Defer Code Execution

Some casks—notably nightlies—have versioned download URLs but are updated so often that they become impractical to keep current with the usual process. For those, we want to dynamically determine `url`.

##### The Problem

In theory, one can write arbitrary Ruby code right in the Cask definition to fetch and construct a disposable URL.

However, this typically involves an HTTP round trip to a landing site, which may take a long time. Because of the way Homebrew Cask loads and parses Casks, it is not acceptable that such expensive operations be performed directly in the body of a Cask definition.

##### Writing the Block

Similar to the `preflight`, `postflight`, `uninstall_preflight`, and `uninstall_postflight` blocks, the `url` stanza offers an optional _block syntax_:

```rb
url "https://handbrake.fr/nightly.php" do |page|
  file_path = page[/href=["']?([^"' >]*Handbrake[._-][^"' >]+\.dmg)["' >]/i, 1]
  file_path ? URI.join(page.url, file_path) : nil
end
```

You can also nest `url do` blocks inside `url do` blocks to follow a chain of URLs.

The block is only evaluated when needed, for example on download time or when auditing a Cask. Inside a block, you may safely do things such as HTTP/S requests that may take a long time to execute. You may also refer to the `@cask` instance variable, and invoke any method available on `@cask`.

The block will be called immediately before downloading; its result value will be assumed to be a `String` (or a pair of a `String` and `Hash` containing parameters) and subsequently used as a download URL.

You can use the `url` stanza with either a direct argument or a block but not with both.

Example for using the block syntax: [vlc-nightly.rb](https://github.com/Homebrew/homebrew-cask-versions/blob/2bf0f13dd49d263ebec0ca56e58ad8458633f789/Casks/vlc-nightly.rb#L5L10)

##### Mixing Additional URL Parameters With the Block Syntax

In rare cases, you might need to set URL parameters like `cookies` or `referer` while also using the block syntax.

This is possible by returning a two-element array as a block result. The first element of the array must be the download URL; the second element must be a `Hash` containing the parameters.

### Stanza: `version`

`version`, while related to the app’s own versioning, doesn’t have to follow it exactly. It is common to change it slightly so it can be [interpolated](https://en.wikipedia.org/wiki/String_interpolation#Ruby_/_Crystal) in other stanzas, usually in `url` to create a Cask that only needs `version` and `sha256` changes when updated. This can be taken further, when needed, with [ruby String methods](https://ruby-doc.org/core/String.html).

For example:

Instead of

```ruby
version "1.2.3"
url "https://example.com/file-version-123.dmg"
```

We can use

```ruby
version "1.2.3"
url "https://example.com/file-version-#{version.delete('.')}.dmg"
```

We can also leverage the power of regular expressions. So instead of

```ruby
version "1.2.3build4"
url "https://example.com/1.2.3/file-version-1.2.3build4.dmg"
```

We can use

```ruby
version "1.2.3build4"
url "https://example.com/#{version.sub(%r{build\d+}, '')}/file-version-#{version}.dmg"
```

#### version :latest

The special value `:latest` is used on casks which:

1. `url` doesn’t contain a version.
2. Having a correct value to `version` is too difficult or impractical, even with our automated systems.

Example: [spotify.rb](https://github.com/Homebrew/homebrew-cask/blob/f56e8ba057687690e26a6502623aa9476ff4ac0e/Casks/spotify.rb#L2)

#### version methods

The examples above can become hard to read, however. Since many of these changes are common, we provide a number of helpers to clearly interpret otherwise obtuse cases:

| Method                   | Input              | Output             |
|--------------------------|--------------------|--------------------|
| `major`                  | `1.2.3-a45,ccdd88` | `1`                |
| `minor`                  | `1.2.3-a45,ccdd88` | `2`                |
| `patch`                  | `1.2.3-a45,ccdd88` | `3-a45`            |
| `major_minor`            | `1.2.3-a45,ccdd88` | `1.2`              |
| `major_minor_patch`      | `1.2.3-a45,ccdd88` | `1.2.3-a45`        |
| `minor_patch`            | `1.2.3-a45,ccdd88` | `2.3-a45`          |
| `before_comma`           | `1.2.3-a45,ccdd88` | `1.2.3-a45`        |
| `after_comma`            | `1.2.3-a45,ccdd88` | `ccdd88`           |
| `dots_to_hyphens`        | `1.2.3-a45,ccdd88` | `1-2-3-a45,ccdd88` |
| `no_dots`                | `1.2.3-a45,ccdd88` | `123-a45,ccdd88`   |

Similar to `dots_to_hyphens`, we provide all logical permutations of `{dots,hyphens,underscores}_to_{dots,hyphens,underscores}`. The same applies to `no_dots` in the form of `no_{dots,hyphens,underscores}`, with an extra `no_dividers` that applies all of those at once.

Finally, there is `csv` that returns an array of comma-separated values. `csv`, `before_comma` and `after_comma` are extra special to allow for otherwise complex cases, and should be used sparingly. There should be no more than two of `,` per `version`.

### Stanza: `zap`

#### `zap` Stanza Purpose

The `zap` stanza describes a more complete uninstallation of files associated with a Cask. The `zap` procedures will never be performed by default, but only if the user uses `--zap` on `uninstall`:

```bash
$ brew uninstall --zap firefox
```

`zap` stanzas may remove:

* Preference files and caches stored within the user’s `~/Library` directory.
* Shared resources such as application updaters. Since shared resources may be removed, other applications may be affected by `brew uninstall --zap`. Understanding that is the responsibility of the end user.

`zap` stanzas should not remove:

* Files created by the user directly.

Appending `--force` to the command will allow you to perform these actions even if the Cask is no longer installed:

```bash
brew uninstall --zap --force firefox
```

#### `zap` Stanza Syntax

The form of `zap` stanza follows the [`uninstall` stanza](#stanza-uninstall). All of the same directives are available. The `trash:` key is preferred over `delete:`.

Example: [dropbox.rb](https://github.com/Homebrew/homebrew-cask/blob/31cd96cc0e00dab1bff74d622e32d816bafd1f6f/Casks/dropbox.rb#L17-L35)

#### `zap` Creation

The simplest method is to use [@nrlquaker's CreateZap](https://github.com/nrlquaker/homebrew-createzap), which can automatically generate the stanza. In a few instances it may fail to pick up anything and manual creation may be required.

Manual creation can be facilitated with:

* Some of the developer tools are already available in Homebrew Cask.
* `sudo find / -iname "*<search item>*"`
* An uninstaller tool such as [AppCleaner](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/appcleaner.rb).
* Inspecting the usual suspects, i.e. `/Library/{'Application Support',LaunchAgents,LaunchDaemons,Frameworks,Logs,Preferences,PrivilegedHelperTools}` and `~/Library/{'Application Support',Caches,Containers,LaunchAgents,Logs,Preferences,'Saved Application State'}`.



---

## Token reference

This section describes the algorithm implemented in the `generate_cask_token` script, and covers detailed rules and exceptions which are not needed in most cases.

* [Purpose](#purpose)
* [Finding the Simplified Name of the Vendor’s Distribution](#finding-the-simplified-name-of-the-vendors-distribution)
* [Converting the Simplified Name To a Token](#converting-the-simplified-name-to-a-token)
* [Cask Filenames](#cask-filenames)
* [Cask Headers](#cask-headers)
* [Cask Token Examples](#cask-token-examples)
* [Tap Specific Cask Token Examples](#tap-specific-cask-token-examples)
* [Token Overlap](#token-overlap)

## Purpose

Software vendors are often inconsistent with their naming. By enforcing strict naming conventions we aim to:

* Prevent duplicate submissions
* Minimize renaming events
* Unambiguously boil down the name of the software into a unique identifier

Details of software names and brands will inevitably be lost in the conversion to a minimal token. To capture the vendor’s full name for a distribution, use the [`name`](#stanza-name) within a Cask. `name` accepts an unrestricted UTF-8 string.

## Finding the Simplified Name of the Vendor’s Distribution

### Simplified Names of Apps

* Start with the exact name of the Application bundle as it appears on disk, such as `Google Chrome.app`.

* If the name uses letters outside A-Z, convert it to ASCII as described in [Converting to ASCII](#converting-to-ascii).

* Remove `.app` from the end.

* Remove from the end: the string “app”, if the vendor styles the name like “Software App.app”. Exception: when “app” is an inseparable part of the name, without which the name would be inherently nonsensical, as in [whatsapp.rb](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/whatsapp.rb).

* Remove from the end: version numbers or incremental release designations such as “alpha”, “beta”, or “release candidate”. Strings which distinguish different capabilities or codebases such as “Community Edition” are currently accepted. Exception: when a number is not an incremental release counter, but a differentiator for a different product from a different vendor, as in [kdiff3.rb](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/kdiff3.rb).

* If the version number is arranged to occur in the middle of the App name, it should also be removed.

* Remove from the end: “Launcher”, “Quick Launcher”.

* Remove from the end: strings such as “Desktop”, “for Desktop”.

* Remove from the end: strings such as “Mac”, “for Mac”, “for OS X”, “macOS”, “for macOS”. These terms are generally added to ported software such as “MAME OS X.app”. Exception: when the software is not a port, and “Mac” is an inseparable part of the name, without which the name would be inherently nonsensical, as in [PlayOnMac.app](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/playonmac.rb).

* Remove from the end: hardware designations such as “for x86”, “32-bit”, “ppc”.

* Remove from the end: software framework names such as “Cocoa”, “Qt”, “Gtk”, “Wx”, “Java”, “Oracle JVM”, etc. Exception: the framework is the product being Casked.

* Remove from the end: localization strings such as “en-US”.

* If the result of that process is a generic term, such as “Macintosh Installer”, try prepending the name of the vendor or developer, followed by a hyphen. If that doesn’t work, then just create the best name you can, based on the vendor’s web page.

* If the result conflicts with the name of an existing Cask, make yours unique by prepending the name of the vendor or developer, followed by a hyphen. Example: [unison.rb](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/unison.rb) and [panic-unison.rb](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/panic-unison.rb).

* Inevitably, there are a small number of exceptions not covered by the rules. Don’t hesitate to [use the forum](https://github.com/Homebrew/discussions/discussions) if you have a problem.

### Converting to ASCII

* If the vendor provides an English localization string, that is preferred. Here are the places it may be found, in order of preference:

  - `CFBundleDisplayName` in the main `Info.plist` file of the app bundle
  - `CFBundleName` in the main `Info.plist` file of the app bundle
  - `CFBundleDisplayName` in `InfoPlist.strings` of an `en.lproj` localization directory
  - `CFBundleName` in `InfoPlist.strings` of an `en.lproj` localization directory
  - `CFBundleDisplayName` in `InfoPlist.strings` of an `English.lproj` localization directory
  - `CFBundleName` in `InfoPlist.strings` of an `English.lproj` localization directory

* When there is no vendor localization string, romanize the name by transliteration or decomposition.

* As a last resort, translate the name of the app bundle into English.

### Simplified Names of `pkg`-based Installers

* The Simplified Name of a `pkg` may be more tricky to determine than that of an App. If a `pkg` installs an App, then use that App name with the rules above. If not, just create the best name you can, based on the vendor’s web page.

### Simplified Names of non-App Software

* Currently, rules for generating a token are not well-defined for Preference Panes, QuickLook plugins, and several other types of software installable by Homebrew Cask. Just create the best name you can, based on the filename on disk or the vendor’s web page. Watch out for duplicates.

  Non-app tokens should become more standardized in the future.

## Converting the Simplified Name To a Token

The token is the primary identifier for a package in our project. It’s the unique string users refer to when operating on the Cask.

To convert the App’s Simplified Name (above) to a token:

* Convert all letters to lower case.
* Expand the `+` symbol into a separated English word: `-plus-`.
* Expand the `@` symbol into a separated English word: `-at-`.
* Spaces become hyphens.
* Underscores become hyphens.
* Middots/Interpuncts become hyphens.
* Hyphens stay hyphens.
* Digits stay digits.
* Delete any character which is not alphanumeric or a hyphen.
* Collapse a series of multiple hyphens into one hyphen.
* Delete a leading or trailing hyphen.

## Cask Filenames

Casks are stored in a Ruby file named after the token, with the file extension `.rb`.

## Cask Headers

The token is also given in the header line for each Cask.

## Cask Token Examples

These illustrate most of the rules for generating a token:

App Name on Disk       | Simplified App Name | Cask Token       | Filename
-----------------------|---------------------|------------------|----------------------
`Audio Hijack Pro.app` | Audio Hijack Pro    | audio-hijack-pro | `audio-hijack-pro.rb`
`VLC.app`              | VLC                 | vlc              | `vlc.rb`
`BetterTouchTool.app`  | BetterTouchTool     | bettertouchtool  | `bettertouchtool.rb`
`LPK25 Editor.app`     | LPK25 Editor        | lpk25-editor     | `lpk25-editor.rb`
`Sublime Text 2.app`   | Sublime Text        | sublime-text     | `sublime-text.rb`

## Tap Specific Cask Token Examples

Cask taps have naming conventions specific to each tap.

[Homebrew/cask-versions](https://github.com/Homebrew/homebrew-cask-versions/blob/HEAD/CONTRIBUTING.md#naming-versions-casks)

[Homebrew/cask-fonts](https://github.com/Homebrew/homebrew-cask-fonts/blob/HEAD/CONTRIBUTING.md#naming-font-casks)

[Homebrew/cask-drivers](https://github.com/Homebrew/homebrew-cask-drivers/blob/HEAD/CONTRIBUTING.md#naming-driver-casks)

# Special Affixes

A few situations require a prefix or suffix to be added to the token.

## Token Overlap

When the token for a new Cask would otherwise conflict with the token of an already existing Cask, the nature of that overlap dictates the token (for possibly both Casks). See [Forks and Apps with Conflicting Names](Acceptable-Casks.md#forks-and-apps-with-conflicting-names) for information on how to proceed.

## Potentially Misleading Name

If the token for a piece of unofficial software that interacts with a popular service would make it look official and the vendor is not authorised to use the name, [a prefix must be added](Acceptable-Casks.md#forks-and-apps-with-conflicting-names) for disambiguation.

In cases where the prefix is ambiguous and would make the app appear official, the `-unofficial` suffix may be used.
