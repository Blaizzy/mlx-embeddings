# Adding Software to Homebrew

Are you missing your favorite software in Homebrew? Then you're the perfect person to resolve this problem.

If you want to add software that is either closed source, or a GUI only program you will want to follow the guide for [Casks](#casks). Otherwise follow the guide for [Formulae](#formulae).
See also: https://docs.brew.sh/Formula-Cookbook#homebrew-terminology

Before you start, please check the open pull requests for [homebrew-core](https://github.com/Homebrew/homebrew-core/pulls) or [homebrew-cask](https://github.com/Homebrew/homebrew-core/pulls), to make sure no one else beat you to the punch.

Next, you will want to go through the [Acceptable Formulae](Acceptable-Formulae.md) or [Acceptable Casks](Acceptable-Casks.md) documentation to determine if the software is an appropriate addition to Homebrew. If you are creating a formula for an alternative version of software already in Homebrew (for example, a major/minor version that significantly differs from the existing version), be sure to read the [Versions](Versions.md) documentation to understand versioned formulae requirements.

If everything checks out, you're ready to get started on a new formula!

## Formulae

### Writing the formula

1. It's a good idea to find existing formulae in Homebrew that have similarities to the software you want to add. This will help you to understand how specific languages, build methods, etc. are typically handled.

1. If you're starting from scratch, the [`brew create` command](Manpage.md#create-options-url) can be used to produce a basic version of your formula. This command accepts a number of options and you may be able to save yourself some work by using an appropriate template option like `--python`.

1. You will now have to work to develop the boilerplate code from `brew create` into a fully-fledged formula. Your main references will be the [Formula Cookbook](Formula-Cookbook.md), similar formulae in Homebrew, and the upstream documentation for your chosen software. Be sure to also take note of the Homebrew documentation for writing [`Python`](Python-for-Formula-Authors.md) and [`Node`](Node-for-Formula-Authors.md) formulae, if applicable.

1. Make sure you write a good test as part of your formula. Refer to the "[Add a test to the formula](Formula-Cookbook.md#add-a-test-to-the-formula)" section of the Cookbook for help with this.

1. Try to install the formula using `brew install --build-from-source <formula>`, where \<formula\> is the name of your formula. If any errors occur, correct your formula and attempt to install it again. The formula should install without errors by the end of this step.

If you're stuck, ask for help on GitHub or [Homebrew/discussions](https://github.com/homebrew/discussions/discussions). The maintainers are very happy to help but we also like to see that you've put effort into trying to find a solution first.

### Testing and auditing the formula

1. Run `brew audit --strict --new-formula --online <formula>` with your formula. If any errors occur, correct them in your formula and run the audit again. The audit should finish without any errors by the end of this step.

1. Run your formula's test using `brew test <formula>` with your formula. Your test should finish without any errors.

### Submitting the formula

You're finally ready to submit your formula to the [homebrew-core](https://github.com/Homebrew/homebrew-core/) repository. If you haven't done this before, you can refer to the [How to Open a Pull Request](How-To-Open-a-Homebrew-Pull-Request.md) documentation for help. Maintainers will review the pull request and provide feedback about any areas that need to be addressed before the formula can be added to Homebrew.

If you've made it this far, congratulations on submitting a Homebrew formula! We appreciate the hard work you put into this and you can take satisfaction in knowing that your work may benefit other Homebrew users as well.

## Casks

**Note**: Before taking the time to craft a new cask, make sure
- it can be accepted by checking the [Rejected Casks FAQ](Acceptable-Casks.md#rejected-casks),
- check if the cask was not [already refused].

[already refused]: https://github.com/Homebrew/homebrew-cask/search?q=is%3Aclosed&type=Issues

### Writing the cask

Making a new Cask is easy. Follow the directions in [Getting Set Up To Contribute](https://github.com/Homebrew/homebrew-cask/blob/HEAD/CONTRIBUTING.md#getting-set-up-to-contribute) to begin.

#### Examples

Here’s a Cask for `shuttle` as an example. Note the `verified` parameter below the `url`, which is needed when [the url and homepage hostnames differ](Cask-Cookbook.md#when-url-and-homepage-domains-differ-add-verified)

```ruby
cask "shuttle" do
  version "1.2.9"
  sha256 "0b80bf62922291da391098f979683e69cc7b65c4bdb986a431e3f1d9175fba20"

  url "https://github.com/fitztrev/shuttle/releases/download/v#{version}/Shuttle.zip",
      verified: "github.com/fitztrev/shuttle/"
  name "Shuttle"
  desc "Simple shortcut menu"
  homepage "https://fitztrev.github.io/shuttle/"

  app "Shuttle.app"

  zap trash: "~/.shuttle.json"
end
```

And here is one for `noisy`. Note that it has an unversioned download (the download `url` does not contain the version number, unlike the example above). It also suppresses the checksum with `sha256 :no_check`, which is necessary because the checksum will change on the same `url` when a new distribution is made available.

```ruby
cask "noisy" do
  version "1.3"
  sha256 :no_check

  url "https://github.com/downloads/jonshea/Noisy/Noisy.zip"
  name "Noisy"
  desc "White noise generator"
  homepage "https://github.com/jonshea/Noisy"

  app "Noisy.app"
end
```

Here is a last example for `airdisplay`, which uses a `pkg` installer to install the application instead of a stand-alone application bundle (`.app`). Note the [`uninstall pkgutil` stanza](Cask-Cookbook.md#uninstall-key-pkgutil), which is needed to uninstall all files which were installed using the installer.

You will also see how to adapt `version` to the download `url`. Use [our custom `version` methods](Cask-Cookbook.md#version-methods) to do so, resorting to the standard [Ruby String methods](https://ruby-doc.org/core/String.html) when they don’t suffice.

```ruby
cask "airdisplay" do
  version "3.4.2,26581"
  sha256 "272d14f33b3a4a16e5e0e1ebb2d519db4e0e3da17f95f77c91455b354bee7ee7"

  url "https://www.avatron.com/updates/software/airdisplay/ad#{version.before_comma.no_dots}.zip"
  name "Air Display"
  desc "Utility for using a tablet as a second monitor"
  homepage "https://avatron.com/applications/air-display/"

  livecheck do
    url "https://www.avatron.com/updates/software/airdisplay/appcast.xml"
    strategy :sparkle
  end

  depends_on macos: ">= :mojave"

  pkg "Air Display Installer.pkg"

  uninstall pkgutil: [
    "com.avatron.pkg.AirDisplay",
    "com.avatron.pkg.AirDisplayHost2",
  ]
end
```

#### Generating a Token for the Cask

The Cask **token** is the mnemonic string people will use to interact with the Cask via `brew install`, etc. The name of the Cask **file** is simply the token with the extension `.rb` appended.

The easiest way to generate a token for a Cask is to run this command:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/generate_cask_token" '/full/path/to/new/software.app'
```

If the software you wish to Cask is not installed, or does not have an associated App bundle, just give the full proper name of the software instead of a pathname:

```bash
$ "$(brew --repository homebrew/cask)/developer/bin/generate_cask_token" 'Google Chrome'
```

If the `generate_cask_token` script does not work for you, see [Cask Token Details](#cask-token-details).

#### The `brew create --cask` Command

Once you know the token, create your Cask with the handy-dandy `brew create --cask` command:

```bash
$ brew create --cask download-url --set-name my-new-cask
```

This will open `$EDITOR` with a template for your new Cask, to be stored in the file `my-new-cask.rb`. Running the `create` command above will get you a template that looks like this:

```ruby
cask "my-new-cask" do
  version ""
  sha256 ""

  url "download-url"
  name ""
  desc ""
  homepage ""

  app ""
end
```

#### Cask Stanzas

Fill in the following stanzas for your Cask:

| name               | value       |
| ------------------ | ----------- |
| `version`          | application version
| `sha256`           | SHA-256 checksum of the file downloaded from `url`, calculated by the command `shasum -a 256 <file>`. Can be suppressed by using the special value `:no_check`. (see [sha256](Cask-Cookbook.md#stanza-sha256))
| `url`              | URL to the `.dmg`/`.zip`/`.tgz`/`.tbz2` file that contains the application.<br />A [`verified` parameter](Cask-Cookbook.md#when-url-and-homepage-domains-differ-add-verified) must be added if the hostnames in the `url` and `homepage` stanzas differ. [Block syntax](Cask-Cookbook.md#using-a-block-to-defer-code-execution) is available for URLs that change on every visit
| `name`             | the full and proper name defined by the vendor, and any useful alternate names (see [Name Stanza Details](Cask-Cookbook.md#stanza-name))
| `desc`             | one-line description of the software (see [Desc Stanza Details](Cask-Cookbook.md#stanza-desc))
| `homepage`         | application homepage; used for the `brew home` command
| `app`              | relative path to an `.app` bundle that should be moved into the `/Applications` folder on installation (see [App Stanza Details](Cask-Cookbook.md#stanza-app))

Other commonly-used stanzas are:

| name               | value       |
| ------------------ | ----------- |
| `livecheck`          | Ruby block describing how to find updates for this Cask (see [Livecheck Stanza Details](Cask-Cookbook.md#stanza-livecheck))
| `pkg`              | relative path to a `.pkg` file containing the distribution (see [Pkg Stanza Details](Cask-Cookbook.md#stanza-pkg))
| `caveats`          | a string or Ruby block providing the user with Cask-specific information at install time (see [Caveats Stanza Details](Cask-Cookbook.md#stanza-caveats))
| `uninstall`        | procedures to uninstall a Cask. Optional unless the `pkg` stanza is used. (see [Uninstall Stanza Details](Cask-Cookbook.md#stanza-uninstall))
| `zap`        		 | additional procedures for a more complete uninstall, including configuration files and shared resources (see [Zap Stanza Details](Cask-Cookbook.md#stanza-zap))

Additional `artifact` stanzas you might need for special use-cases can be found [here](Cask-Cookbook.md#at-least-one-artifact-stanza-is-also-required). Even more special-use stanzas are listed at [Optional Stanzas](Cask-Cookbook.md#optional-stanzas).

#### Cask Token Details

If a token conflicts with an already-existing Cask, authors should manually make the new token unique by prepending the vendor name. Example: [unison.rb](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/unison.rb) and [panic-unison.rb](https://github.com/Homebrew/homebrew-cask/blob/HEAD/Casks/panic-unison.rb).

If possible, avoid creating tokens which differ only by the placement of hyphens.

To generate a token manually, or to learn about exceptions for unusual cases, see [token_reference.md](Cask-Cookbook.md#token-reference).

#### Archives With Subfolders

When a downloaded archive expands to a subfolder, the subfolder name must be included in the `app` value.

Example:

1. Texmaker is downloaded to the file `TexmakerMacosxLion.zip`.
2. `TexmakerMacosxLion.zip` unzips to a folder called `TexmakerMacosxLion`.
3. The folder `TexmakerMacosxLion` contains the application `texmaker.app`.
4. So, the `app` stanza should include the subfolder as a relative path:

  ```ruby
  app "TexmakerMacosxLion/texmaker.app"
  ```

### Testing and auditing the cask

Give it a shot with:

```bash
export HOMEBREW_NO_AUTO_UPDATE=1
brew install my-new-cask
```

Did it install? If something went wrong, edit your Cask with `brew edit my-new-cask` to fix it.

Test also if the uninstall works successfully:

```bash
brew uninstall my-new-cask
```

If everything looks good, you’ll also want to make sure your Cask passes audit with:

```bash
brew audit --new-cask my-new-cask
```

You should also check stylistic details with `brew style`:

```bash
brew style --fix my-new-cask
```

Keep in mind all of these checks will be made when you submit your PR, so by doing them in advance you’re saving everyone a lot of time and trouble.

If your application and Homebrew Cask do not work well together, feel free to [file an issue](https://github.com/Homebrew/homebrew-cask#reporting-bugs) after checking out open issues.

### Submitting the cask

#### Finding a Home For Your Cask

See the [Acceptable Casks documentation](Acceptable-Casks.md#finding-a-home-for-your-cask).

Hop into your Tap and check to make sure your new Cask is there:

```bash
$ cd "$(brew --repository)"/Library/Taps/homebrew/homebrew-cask
$ git status
# On branch master
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#
#       Casks/my-new-cask.rb
```

So far, so good. Now make a feature branch `my-new-cask-branch` that you’ll use in your pull request:

```bash
$ git checkout -b my-new-cask-branch
Switched to a new branch 'my-new-cask-branch'
```

Stage your Cask with:

```bash
$ git add Casks/my-new-cask.rb
```

You can view the changes that are to be committed with:

```bash
$ git diff --cached
```

Commit your changes with:

```bash
$ git commit -v
```

#### Commit Messages

For any git project, some good rules for commit messages are:

* The first line is commit summary, 50 characters or less,
* Followed by an empty line,
* Followed by an explanation of the commit, wrapped to 72 characters.

See [a note about git commit messages](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) for more.

The first line of a commit message becomes the **title** of a pull request on GitHub, like the subject line of an email. Including the key info in the first line will help us respond faster to your pull.

For Cask commits in the Homebrew Cask project, we like to include the Application name, version number, and purpose of the commit in the first line.

Examples of good, clear commit summaries:

* `Add Transmission.app v1.0`
* `Upgrade Transmission.app to v2.82`
* `Fix checksum in Transmission.app Cask`
* `Add CodeBox Latest`

Examples of difficult, unclear commit summaries:

* `Upgrade to v2.82`
* `Checksum was bad`

#### Pushing

Push your changes from the branch `my-new-cask-branch` to your GitHub account:

```bash
$ git push {{my-github-username}} my-new-cask-branch
```

If you are using [GitHub two-factor authentication](https://help.github.com/articles/about-two-factor-authentication/) and set your remote repository as HTTPS you will need to set up a personal access token and use that instead of your password. Further information [here](https://help.github.com/articles/https-cloning-errors/#provide-access-token-if-2fa-enabled).

#### Filing a Pull Request on GitHub

##### a) suggestion from git push

The `git push` command prints a suggestion to create a pull request:

```
remote: Create a pull request for 'new-cask-cask' on GitHub by visiting:
remote:      https://github.com/{{my-github-username}}/homebrew-cask/pull/new/my-new-cask-branch
```

##### b) use suggestion at Github website

Now go to the [`homebrew-cask` GitHub repository](https://github.com/Homebrew/homebrew-cask). GitHub will often show your `my-new-cask-branch` branch with a handy button to `Compare & pull request`.

##### c) manually create a pull request at Github website

Otherwise, click the `New pull request` button and choose to `compare across forks`. The base fork should be `Homebrew/homebrew-cask @ master`, and the head fork should be `my-github-username/homebrew-cask @ my-new-cask-branch`. You can also add any further comments to your pull request at this stage.

##### Congratulations!

You are done now, and your Cask should be pulled in or otherwise noticed in a while. If a maintainer suggests some changes, just make them on the `my-new-cask-branch` branch locally and [push](#pushing).

### Cleaning up

After your Pull Request is submitted, you should get yourself back onto `master`, so that `brew update` will pull down new Casks properly:

```bash
cd "$(brew --repository)"/Library/Taps/homebrew/homebrew-cask
git checkout master
```

if you set the variable `HOMEBREW_NO_AUTO_UPDATE` then clean it up with:

```bash
unset HOMEBREW_NO_AUTO_UPDATE
```

