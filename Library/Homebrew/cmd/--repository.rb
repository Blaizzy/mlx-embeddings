# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def __repository_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--repository`, `--repo` [<user>`/`<repo>]

        Display where Homebrew's `.git` directory is located.

        If <user>`/`<repo> are provided, display where tap <user>`/`<repo>'s directory is located.
      EOS
    end
  end

  def __repository
    args = __repository_args.parse

    if args.no_named?
      puts HOMEBREW_REPOSITORY
    else
      puts args.named.map { |tap| Tap.fetch(tap).path }
    end
  end
end
