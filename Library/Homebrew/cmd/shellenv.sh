#:  * `shellenv`
#:
#:  Print export statements. When run in a shell, this installation of Homebrew will be added to your `PATH`, `MANPATH`, and `INFOPATH`.
#:
#:  The variables `HOMEBREW_PREFIX`, `HOMEBREW_CELLAR` and `HOMEBREW_REPOSITORY` are also exported to avoid querying them multiple times.
#:  Consider adding evaluation of this command's output to your dotfiles (e.g. `~/.profile` or `~/.zprofile`) with: `eval $(brew shellenv)`

homebrew-shellenv() {
  case "$SHELL" in
    */fish)
      echo "set -gx HOMEBREW_PREFIX \"$HOMEBREW_PREFIX\";"
      echo "set -gx HOMEBREW_CELLAR \"$HOMEBREW_CELLAR\";"
      echo "set -gx HOMEBREW_REPOSITORY \"$HOMEBREW_REPOSITORY\";"
      echo "set -g fish_user_paths \"$HOMEBREW_PREFIX/bin\" \"$HOMEBREW_PREFIX/sbin\" \$fish_user_paths;"
      echo "set -q MANPATH; or set MANPATH ''; set -gx MANPATH \"$HOMEBREW_PREFIX/share/man\" \$MANPATH;"
      echo "set -q INFOPATH; or set INFOPATH ''; set -gx INFOPATH \"$HOMEBREW_PREFIX/share/info\" \$INFOPATH;"
      ;;
    */csh|*/tcsh)
      echo "setenv HOMEBREW_PREFIX $HOMEBREW_PREFIX;"
      echo "setenv HOMEBREW_CELLAR $HOMEBREW_CELLAR;"
      echo "setenv HOMEBREW_REPOSITORY $HOMEBREW_REPOSITORY;"
      echo "setenv PATH $HOMEBREW_PREFIX/bin:$HOMEBREW_PREFIX/sbin:\$PATH;"
      echo "setenv MANPATH $HOMEBREW_PREFIX/share/man:\$MANPATH;"
      echo "setenv INFOPATH $HOMEBREW_PREFIX/share/info:\$INFOPATH;"
      ;;
    *)
      echo "export HOMEBREW_PREFIX=\"$HOMEBREW_PREFIX\";"
      echo "export HOMEBREW_CELLAR=\"$HOMEBREW_CELLAR\";"
      echo "export HOMEBREW_REPOSITORY=\"$HOMEBREW_REPOSITORY\";"
      echo "export PATH=\"$HOMEBREW_PREFIX/bin:$HOMEBREW_PREFIX/sbin\${PATH+:\$PATH}\";"
      echo "export MANPATH=\"$HOMEBREW_PREFIX/share/man\${MANPATH+:\$MANPATH}:\";"
      echo "export INFOPATH=\"$HOMEBREW_PREFIX/share/info\${INFOPATH+:\$INFOPATH}\";"
      ;;
  esac
}
