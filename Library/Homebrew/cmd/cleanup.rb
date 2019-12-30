# frozen_string_literal: true

require "cleanup"
require "cli/parser"

module Homebrew
  module_function

  def cleanup_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `cleanup` [<options>] [<formula>|<cask>]

        Remove stale lock files and outdated downloads for all formulae and casks,
        and remove old versions of installed formulae. If arguments are specified,
        only do this for the given formulae and casks.
      EOS
      flag   "--prune=",
             description: "Remove all cache files older than specified <days>."
      switch "-n", "--dry-run",
             description: "Show what would be removed, but do not actually remove anything."
      switch "-s",
             description: "Scrub the cache, including downloads for even the latest versions. "\
                          "Note downloads for any installed formulae or casks will still not be deleted. "\
                          "If you want to delete those too: `rm -rf \"$(brew --cache)\"`"
      switch "--prune-prefix",
             description: "Only prune the symlinks and directories from the prefix and remove no other files."
      switch :verbose
      switch :debug
    end
  end

  def cleanup
    cleanup_args.parse

    cleanup = Cleanup.new(*args.remaining, dry_run: args.dry_run?, scrub: args.s?, days: args.prune&.to_i)
    if args.prune_prefix?
      cleanup.prune_prefix_symlinks_and_directories
      return
    end

    cleanup.clean!

    unless cleanup.disk_cleanup_size.zero?
      disk_space = disk_usage_readable(cleanup.disk_cleanup_size)
      if args.dry_run?
        ohai "This operation would free approximately #{disk_space} of disk space."
      else
        ohai "This operation has freed approximately #{disk_space} of disk space."
      end
    end

    return if cleanup.unremovable_kegs.empty?

    ofail <<~EOS
      Could not cleanup old kegs! Fix your permissions on:
        #{cleanup.unremovable_kegs.join "\n  "}
    EOS
  end
end
