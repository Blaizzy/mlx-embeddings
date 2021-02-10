# FAQ

## Is there a glossary of terms around?
All your terminology needs can be [found here](Formula-Cookbook.md#homebrew-terminology).

## How do I update my local packages?
First update the formulae and Homebrew itself:

    brew update

You can now find out what is outdated with:

    brew outdated

Upgrade everything with:

    brew upgrade

Or upgrade a specific formula with:

    brew upgrade <formula>

## How do I stop certain formulae from being updated?
To stop something from being updated/upgraded:

    brew pin <formula>

To allow that formulae to update again:

    brew unpin <formula>

Note that pinned, outdated formulae that another formula depends on need to be upgraded when required, as we do not allow formulae to be built against outdated versions. If this is not desired, you can instead `brew extract` to [maintain your own copy of the formula in a tap](How-to-Create-and-Maintain-a-Tap.md).

## How do I uninstall Homebrew?
To uninstall Homebrew, run the [uninstall script from the Homebrew/install repository](https://github.com/homebrew/install#uninstall-homebrew).

## How can I keep old versions of a formula when upgrading?
Homebrew automatically uninstalls old versions of a formula after that formula is upgraded with `brew upgrade`, and periodically performs additional cleanup every 30 days.

To __disable__ automatic `brew cleanup`:

    export HOMEBREW_NO_INSTALL_CLEANUP=1

When automatic `brew cleanup` is disabled, if you uninstall a formula, it will only remove the latest version you have installed. It will not remove all versions of the formula that you may have installed in the past. Homebrew will continue to attempt to install the newest version it knows about when you run `brew upgrade`. This can be surprising.

In this case, to remove a formula entirely, you may run `brew uninstall --force <formula>`. Be careful as this is a destructive operation.

## Why does `brew upgrade <formula>` also upgrade a bunch of other stuff?
Homebrew doesn't support arbitrary mixing and matching of formula versions, so everything a formula depends on, and everything that depends on it in turn, needs to be upgraded to the latest version as that's the only combination of formulae we test. As a consequence any given `upgrade` or `install` command can upgrade many other (seemingly unrelated) formulae, if something important like `python` or `openssl` also needed an upgrade.

## Where does stuff get downloaded?

    brew --cache

Which is usually: `~/Library/Caches/Homebrew`

## My Mac `.app`s don’t find `/usr/local/bin` utilities!
GUI apps on macOS don’t have `/usr/local/bin` in their `PATH` by default. If you're on Mountain Lion or later, you can fix this by running `sudo launchctl config user path "/usr/local/bin:$PATH"` and then rebooting, as documented in `man launchctl`. Note that this sets the launchctl `PATH` for *all users*. For earlier versions of macOS, see [this page](https://developer.apple.com/legacy/library/qa/qa1067/_index.html).

## How do I contribute to Homebrew?
Read our [contribution guidelines](https://github.com/Homebrew/brew/blob/HEAD/CONTRIBUTING.md#contributing-to-homebrew).

## Why do you compile everything?
Homebrew provides pre-compiled versions for many formulae. These
pre-compiled versions are referred to as [bottles](Bottles.md) and are available
at <https://bintray.com/homebrew/bottles>.

If available, bottled binaries will be used by default except under the
following conditions:

* Options were passed to the install command, i.e. `brew install <formula>`
will use a bottled version of the formula, but
`brew install --enable-bar <formula>` will trigger a source build.
* The `--build-from-source` option is invoked.
* The machine is not running a supported version of macOS as all
bottled builds are generated only for supported macOS versions.
* Homebrew is installed to a prefix other than the standard
`/usr/local` (although some bottles support this).

We aim to bottle everything.

## How do I get a formula from someone else’s branch?

```sh
brew install hub
brew update
cd $(brew --repository)
hub pull someone_else
```

## Why should I install Homebrew in the default location?

Homebrew's pre-built binary packages (known as [bottles](Bottles.md)) of many packages can only be used if you install in the default installation prefix, otherwise they have to be built from source. Building from source takes a long time, is prone to fail, and is not supported. Do yourself a favour and install to the default prefix so that you can use our pre-built binary packages. The default prefix is `/usr/local` for macOS on Intel, `/opt/homebrew` for macOS on ARM, and `/home/linuxbrew/.linuxbrew` for Linux. *Pick another prefix at your peril!*

## Why does Homebrew prefer I install to `/usr/local`?

1.  **It’s easier**<br>`/usr/local/bin` is already in your
    `PATH`.
2.  **It’s easier**<br>Tons of build scripts break if their dependencies
    aren’t in either `/usr` or `/usr/local`. We
    fix this for Homebrew formulae (although we don’t always test for
    it), but you’ll find that many RubyGems and Python setup scripts
    break which is something outside our control.
3.  **It’s safe**<br>Apple has assigned this directory for non-system utilities. This means
    there are no files in `/usr/local` by default, so there
    is no need to worry about messing up existing or system tools.

**If you plan to install gems that depend on formulae then save yourself a bunch of hassle and install to `/usr/local`!**

It is not always straightforward to tell `gem` to look in non-standard directories for headers and libraries. If you choose `/usr/local`, many things will "just work".

## Why is the default installation prefix `/home/linuxbrew/.linuxbrew` on Linux?

The prefix `/home/linuxbrew/.linuxbrew` was chosen so that users without admin access can ask an admin to create a `linuxbrew` role account and still benefit from precompiled binaries. If you do not yourself have admin privileges, consider asking your admin staff to create a `linuxbrew` role account for you with home directory `/home/linuxbrew`.

## Why does Homebrew say sudo is bad?
**tl;dr** Sudo is dangerous, and you installed TextMate.app without sudo
anyway.

Homebrew refuses to work using sudo.

You should only ever sudo a tool you trust. Of course, you can trust Homebrew
😉 But do you trust the multi-megabyte Makefile that Homebrew runs? Developers
often understand C++ far better than they understand make syntax. It’s too high
a risk to sudo such stuff. It could modify (or upload) any files on your
system. And indeed, we’ve seen some build scripts try to modify `/usr` even when
the prefix was specified as something else entirely.

We use the macOS sandbox to stop this but this doesn't work when run as the `root` user (which also has read and write access to almost everything on the system).

Did you `chown root /Applications/TextMate.app`? Probably
not. So is it that important to `chown root wget`?

If you need to run Homebrew in a multi-user environment, consider
creating a separate user account especially for use of Homebrew.

## Why isn’t a particular command documented?

If it’s not in `man brew`, it’s probably an external command. These are documented [here](External-Commands.md).

## Why haven’t you merged my pull request?
If it’s been a while, bump it with a “bump” comment. Sometimes we miss requests and there are plenty of them. Maybe we were thinking on something. It will encourage consideration. In the meantime if you could rebase the pull request so that it can be cherry-picked more easily we will love you for a long time.

## Can I edit formulae myself?
Yes! It’s easy! Just `brew edit <formula>`. You don’t have to submit modifications back to `homebrew/core`, just edit the formula as you personally need it and `brew install <formula>`. As a bonus `brew update` will merge your changes with upstream so you can still keep the formula up-to-date **with** your personal modifications!

## Can I make new formulae?
Yes! It’s easy! Just `brew create URL`. Homebrew will then open the formula in
`EDITOR` so you can edit it, but it probably already installs; try it: `brew
install <formula>`. If you encounter any issues, run the command with the
`--debug` switch like so: `brew install --debug <formula>`, which drops you
into a debugging shell.

If you want your new formula to be part of `homebrew/core` or want
to learn more about writing formulae, then please read the [Formula Cookbook](Formula-Cookbook.md).

## Can I install my own stuff to `/usr/local`?
Yes, `brew` is designed to not get in your way so you can use it how you
like.

Install your own stuff, but be aware that if you install common
libraries like libexpat yourself, it may cause trouble when trying to
build certain Homebrew formula. As a result `brew doctor` will warn you
about this.

Thus it’s probably better to install your own stuff to the Cellar and
then `brew link` it. Like so:

```sh
$ cd foo-0.1
$ brew diy
./configure --prefix=/usr/local/Cellar/foo/0.1
$ ./configure --prefix=/usr/local/Cellar/foo/0.1
[snip]
$ make && make install
$ brew link foo
Linking /usr/local/Cellar/foo/0.1… 17 symlinks created
```

## Why was a formula deleted or disabled?
Use `brew log <formula>` to find out! Likely because it had [unresolved issues](Acceptable-Formulae.md) and/or [our analytics](Analytics.md) identified it was not widely used.

For disabled and deprecated formulae, running `brew info <formula>` will also provide an explanation.

## Homebrew is a poor name, it's too generic, why was it chosen?
Homebrew's creator @mxcl was too concerned with the beer theme and didn't consider that the project may actually prove popular. By the time Max realised that it was popular, it was too late. However, today, the first Google hit for "homebrew" is not beer related 😉

## What does "keg-only" mean?
It means the formula is installed only into the Cellar and is not linked into `/usr/local`. This means most tools will not find it. You can see why a formula was installed as keg-only, and instructions to include it in your `PATH`, by running `brew info <formula>`.

You can still link in the formula if you need to with `brew link <formula>`, though this can cause unexpected behaviour if you are shadowing macOS software.

## How can I specify different configure arguments for a formula?
`brew edit <formula>` and edit the formula. Currently there is no
other way to do this.


## The app can’t be opened because it is from an unidentified developer
Chances are that certain apps will give you a popup message like this:

<img src="https://i.imgur.com/CnEEATG.png" width="532" alt="Gatekeeper message">

This is a [security feature from Apple](https://support.apple.com/en-us/HT202491). The single most important thing to know is that **you can allow individual apps to be exempt from that feature.** This allows the app to run while the rest of the system remains under protection.

**Always leave system-wide protection enabled,** and disable it only for specific apps as needed.

If you are sure you want to trust the app, you can disable protection for that app by right-clicking its icon and choosing `Open`:

<img src="https://i.imgur.com/69xc2WK.png" width="312" alt="Right-click the app and choose Open">

Finally, click the `Open` button if you want macOS to permanently allow the app to run on this Mac. **Don’t do this unless you’re sure you trust the app.**

<img src="https://i.imgur.com/xppa4Qv.png" width="532" alt="Gatekeeper message">

Alternatively, you may provide the [`--no-quarantine` flag](https://github.com/Homebrew/homebrew-cask/blob/HEAD/USAGE.md#options) at install time to not add this feature to a specific app.


## Why some apps aren’t included in `upgrade`
After running `brew upgrade`, you may notice some casks you think should be upgrading, aren’t.

As you’re likely aware, a lot of macOS software can upgrade itself:

<img src="https://upload.wikimedia.org/wikipedia/commons/c/c0/Sparkle_Test_App_Software_Update.png" width="532" alt="Sparkle update window">

That could cause conflicts when used in tandem with Homebrew Cask’s `upgrade` mechanism.

If you upgrade software through it’s built-in mechanism, that happens without Homebrew Cask’s knowledge so both versions get out of sync. If you then upgraded through Homebrew Cask and we have a lower version on the software on record, you’d get a downgrade.

There are a few ideas to fix this problem:

* Try to prevent the software’s automated updates. That won’t be a universal solution and may cause it to break. Most software on Homebrew Cask is closed-source, so we’d be guessing. This is also why pinning casks to a version isn’t available.
* Try to extract the installed software’s version and compare it to the cask, deciding what to do at that time. That’s a complicated solution that breaks other parts of our methodology, such as using versions to interpolate in `url`s (a definite win for maintainability). That solution also isn’t universal, as many software developers are inconsistent in their versioning schemes (and app bundles are meant to have two version strings) and it doesn’t work for all types of software we support.

So we let software be. Installing it with Homebrew Cask should make it behave the same as if you had installed it manually. But we also want to support software that does not auto-upgrade, so we add [`auto_updates true`](https://github.com/Homebrew/homebrew-cask/blob/62c0495b254845a481dacac6ea7c8005e27a3fb0/Casks/alfred.rb#L10) to casks of software that can do it, which excludes them from `brew upgrade`.

Casks which use [`version :latest`](https://github.com/Homebrew/homebrew-cask/blob/HEAD/doc/cask_language_reference/stanzas/version.md#version-latest) are also excluded, because we have no way to track the version they’re in. It helps to ask the developers of such software to provide versioned releases (i.e. have the version in the path of the download `url`).

If you still want to force software to be upgraded via Homebrew Cask, you can:

* Reference it specifically in the `upgrade` command: `brew upgrade {{cask_name}}`.
* Use the `--greedy` flag: `brew upgrade --greedy`.

Refer to the `upgrade` section of the `brew` manual page by running `man -P 'less --pattern "^ {3}upgrade"' brew`.