# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def __version_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--version`

        Print the version numbers of Homebrew, Homebrew/homebrew-core and Homebrew/homebrew-cask
        (if tapped) to standard output.
      EOS
      max_named 0
    end
  end

  def __version
    __version_args.parse

    puts "Homebrew #{HOMEBREW_VERSION}"
    puts "#{CoreTap.instance.full_name} #{CoreTap.instance.version_string}"
    puts "#{Tap.default_cask_tap.full_name} #{Tap.default_cask_tap.version_string}" if Tap.default_cask_tap.installed?
  end
end
