# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def untap_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `untap` <tap>

        Remove a tapped formula repository.
      EOS
      switch :debug
    end
  end

  def untap
    untap_args.parse

    raise UsageError, "This command requires a tap argument from `brew tap`'s list" if args.remaining.empty?

    ARGV.named.each do |tapname|
      tap = Tap.fetch(tapname)
      odie "Untapping #{tap} is not allowed" if tap.core_tap?

      tap.uninstall
    end
  end
end
