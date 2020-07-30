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

      min_named 1
    end
  end

  def untap
    args = untap_args.parse

    args.named.each do |tapname|
      tap = Tap.fetch(tapname)
      odie "Untapping #{tap} is not allowed" if tap.core_tap?

      tap.uninstall
    end
  end
end
