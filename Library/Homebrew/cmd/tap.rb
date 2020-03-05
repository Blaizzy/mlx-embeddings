# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def tap_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `tap` [<options>] <user>`/`<repo> [<URL>]

        Tap a formula repository.

        If no arguments are provided, list all installed taps.

        With <URL> unspecified, tap a formula repository from GitHub using HTTPS.
        Since so many taps are hosted on GitHub, this command is a shortcut for
        `brew tap` <user>`/`<repo> `https://github.com/`<user>`/homebrew-`<repo>.

        With <URL> specified, tap a formula repository from anywhere, using
        any transport protocol that `git`(1) handles. The one-argument form of `tap`
        simplifies but also limits. This two-argument command makes no
        assumptions, so taps can be cloned from places other than GitHub and
        using protocols other than HTTPS, e.g. SSH, git, HTTP, FTP(S), rsync.
      EOS
      switch "--full",
             description: "Convert a shallow clone to a full clone without untapping. By default, taps are no "\
                          "longer cloned as shallow clones."
      switch "--force-auto-update",
             description: "Auto-update tap even if it is not hosted on GitHub. By default, only taps "\
                          "hosted on GitHub are auto-updated (for performance reasons)."
      switch "--repair",
             description: "Migrate tapped formulae from symlink-based to directory-based structure."
      switch "--list-pinned",
             description: "List all pinned taps."
      switch :quiet,
             description: "Suppress any warnings."
      switch :debug
      max_named 2
    end
  end

  def tap
    tap_args.parse

    if args.repair?
      Tap.each(&:link_completions_and_manpages)
    elsif args.list_pinned?
      puts Tap.select(&:pinned?).map(&:name)
    elsif args.no_named?
      puts Tap.names
    else
      tap = Tap.fetch(args.named.first)
      begin
        tap.install clone_target:      args.named.second,
                    force_auto_update: force_auto_update?,
                    quiet:             args.quiet?
      rescue TapRemoteMismatchError => e
        odie e
      rescue TapAlreadyTappedError
        nil
      end
    end
  end

  def force_auto_update?
    # if no relevant flag is present, return nil, meaning "no change"
    true if args.force_auto_update?
  end
end
