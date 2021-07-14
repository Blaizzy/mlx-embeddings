# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def developer_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Control Homebrew's developer mode. When developer mode is enabled,
        `brew update` will update Homebrew to the latest commit on the `master`
        branch instead of the latest stable version.

        `brew developer` [`state`]:
        Display the current state of Homebrew's developer mode.

        `brew developer` (`on`|`off`):
        Turn Homebrew's developer mode on or off respectively.
      EOS

      named_args %w[state on off], max: 1
    end
  end

  def developer
    args = developer_args.parse

    case args.named.first
    when nil, "state"
      if Homebrew::EnvConfig.developer?
        puts "Developer mode is enabled because #{Tty.bold}HOMEBREW_DEVELOPER#{Tty.reset} it set."
      elsif Homebrew::Settings.read("devcmdrun") == "true"
        puts "Developer mode is enabled."
      else
        puts "Developer mode is disabled."
      end
    when "on"
      Homebrew::Settings.write "devcmdrun", true
    when "off"
      Homebrew::Settings.delete "devcmdrun"
      if Homebrew::EnvConfig.developer?
        puts "To fully disable developer mode, you must unset #{Tty.bold}HOMEBREW_DEVELOPER#{Tty.reset}."
      end
    else
      raise UsageError, "unknown subcommand: #{args.named.first}"
    end
  end
end
