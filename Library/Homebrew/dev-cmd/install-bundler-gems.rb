# typed: true
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def install_bundler_gems_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Install Homebrew's Bundler gems.
      EOS
      comma_array "--groups=",
                  description: "Installs the specified comma-separated list of gem groups (default: last used)."

      named_args :none
    end
  end

  def install_bundler_gems
    args = install_bundler_gems_args.parse

    # Clear previous settings. We want to fully replace - not append.
    Homebrew::Settings.delete(:gemgroups) if args.groups

    Homebrew.install_bundler_gems!(groups: args.groups || [])
  end
end
