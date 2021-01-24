# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def __repository_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display where Homebrew's git repository is located.

        If <user>`/`<repo> are provided, display where tap <user>`/`<repo>'s directory is located.
      EOS

      named_args :tap
    end
  end

  def __repository
    args = __repository_args.parse

    if args.no_named?
      puts HOMEBREW_REPOSITORY
    else
      puts args.named.to_taps.map(&:path)
    end
  end
end
