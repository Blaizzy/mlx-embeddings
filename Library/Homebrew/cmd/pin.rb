# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  module_function

  def pin_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pin` <formula>

        Pin the specified <formula>, preventing them from being upgraded when
        issuing the `brew upgrade` <formula> command. See also `unpin`.
      EOS
      switch :debug
    end
  end

  def pin
    pin_args.parse

    raise FormulaUnspecifiedError if args.remaining.empty?

    Homebrew.args.resolved_formulae.each do |f|
      if f.pinned?
        opoo "#{f.name} already pinned"
      elsif !f.pinnable?
        onoe "#{f.name} not installed"
      else
        f.pin
      end
    end
  end
end
