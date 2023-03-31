# typed: true
# frozen_string_literal: true

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
      comma_array "--groups",
                  description: "Installs the specified comma-separated list of gem groups (default: last used)."

      named_args :none
    end
  end

  def install_bundler_gems
    args = install_bundler_gems_args.parse

    groups = args.groups

    # Clear previous settings. We want to fully replace - not append.
    Homebrew::Settings.delete(:gemgroups) if groups

    groups ||= []
    groups |= VALID_GEM_GROUPS if groups.delete("all")

    Homebrew.install_bundler_gems!(groups: groups)
  end
end
