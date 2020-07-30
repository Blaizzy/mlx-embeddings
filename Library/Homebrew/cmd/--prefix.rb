# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def __prefix_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--prefix` [<formula>]

        Display Homebrew's install path. *Default:* `/usr/local` on macOS and
        `/home/linuxbrew/.linuxbrew` on Linux.

        If <formula> is provided, display the location in the Cellar where <formula>
        is or would be installed.
      EOS
    end
  end

  def __prefix
    args = __prefix_args.parse

    if args.no_named?
      puts HOMEBREW_PREFIX
    else
      puts args.resolved_formulae.map { |f|
        f.opt_prefix.exist? ? f.opt_prefix : f.installed_prefix
      }
    end
  end
end
