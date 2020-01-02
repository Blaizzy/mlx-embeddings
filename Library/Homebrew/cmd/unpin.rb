# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  module_function

  def unpin_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `unpin` <formula>

        Unpin <formula>, allowing them to be upgraded by `brew upgrade` <formula>.
        See also `pin`.
      EOS
      switch :verbose
      switch :debug
    end
  end

  def unpin
    unpin_args.parse

    raise FormulaUnspecifiedError if args.remaining.empty?

    Homebrew.args.resolved_formulae.each do |f|
      if f.pinned?
        f.unpin
      elsif !f.pinnable?
        onoe "#{f.name} not installed"
      else
        opoo "#{f.name} not pinned"
      end
    end
  end
end
