require "keg"
require "cli_parser"
require "cleanup"

module Homebrew
  module_function

  def prune_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `prune` [<options>]

        Deprecated. Use `brew cleanup` instead.
      EOS
      switch "-n", "--dry-run",
        description: "Show what would be removed, but do not actually remove anything."
      switch :verbose
      switch :debug
      hide_from_man_page!
    end
  end

  def prune
    prune_args.parse

    odisabled("'brew prune'", "'brew cleanup'")
  end
end
