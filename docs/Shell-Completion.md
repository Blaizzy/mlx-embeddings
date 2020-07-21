# Homebrew Shell Completion

Homebrew comes with completion definitions for the `brew` command. Some packages also provide completion definitions for their own programs.

`zsh`, `bash` and `fish` are currently supported.

You must configure your shell to enable its completion support. This is because the Homebrew-managed completions are stored under `HOMEBREW_PREFIX` which your system shell may not be aware of, and since it is difficult to automatically configure `bash` and `zsh` completions in a robust manner, the Homebrew installer does not do it for you.

## Configuring Completions in `bash`

To make Homebrew's completions available in `bash`, you must source the definitions as part of your shell's startup. Add the following to your `~/.bash_profile` (or, if it doesn't exist, `~/.profile):

```sh
if type brew &>/dev/null; then
  HOMEBREW_PREFIX="$(brew --prefix)"
  if [[ -r "${HOMEBREW_PREFIX}/etc/profile.d/bash_completion.sh" ]]; then
    source "${HOMEBREW_PREFIX}/etc/profile.d/bash_completion.sh"
  else
    for COMPLETION in "${HOMEBREW_PREFIX}/etc/bash_completion.d/"*; do
      [[ -r "$COMPLETION" ]] && source "$COMPLETION"
    done
  fi
fi
```

Should you later install the `bash-completion` formula, this will automatically use its initialization script to read the completions files.
When installed, the `bash-completion` formula also runs `${HOMEBREW_PREFIX}/etc/profile.d/bash_completion.sh` and all files in the `bash_completion.d` directory. This is done by adding a line to your `.bash_profile` which is given in the Caveats section upon the installation of `bash-completion`

As both Homebrew's completion code given above and the Caveats line do the same thing, it is recommended to either not add the Caveats line or to comment the line out because Homebrew's completion code works even without installing the `bash-completion` formula.

If you are using a version of `bash` newer than version 4.1 (like Homebrew's `bash`), we recommended you use the `bash-completion@2` formula instead because it is newer and has better performance. You can check the version of `bash` you have by running `bash --version`. MacOS ships with version 3.2.


## Configuring Completions in `zsh`

To make Homebrew's completions available in `zsh`, you must get the Homebrew-managed zsh site-functions on your `FPATH` before initialising `zsh`'s completion facility. Add the following to your `~/.zshrc` file:

```sh
if type brew &>/dev/null; then
  FPATH=$(brew --prefix)/share/zsh/site-functions:$FPATH

  autoload -Uz compinit
  compinit
fi
```

This must be done before `compinit` is called. Note that if you are using Oh My Zsh, it will call `compinit` for you, so this must be done before you call `oh-my-zsh.sh`.

You may also need to forcibly rebuild `zcompdump`:

```sh
  rm -f ~/.zcompdump; compinit
```

Additionally, if you receive "zsh compinit: insecure directories" warnings when attempting to load these completions, you may need to run this:

```sh
  chmod go-w "$(brew --prefix)/share"
```

## Configuring Completions in `fish`

No configuration is needed if you're using Homebrew's `fish`. Friendly!

If your `fish` is from somewhere else, add the following to your `~/.config/fish/config.fish`:

```sh
if test -d (brew --prefix)"/share/fish/completions"
    set -gx fish_complete_path $fish_complete_path (brew --prefix)/share/fish/completions
end

if test -d (brew --prefix)"/share/fish/vendor_completions.d"
    set -gx fish_complete_path $fish_complete_path (brew --prefix)/share/fish/vendor_completions.d
end
```
