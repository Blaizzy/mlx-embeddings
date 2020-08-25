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

      min_named :formula
    end
  end

  def pin
    args = pin_args.parse

    args.named.to_resolved_formulae.each do |f|
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
