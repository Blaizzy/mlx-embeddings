# typed: true
# frozen_string_literal: true

require "readall"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def readall_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Import all items from the specified <tap>, or from all installed taps if none is provided.
        This can be useful for debugging issues across all items when making
        significant changes to `formula.rb`, testing the performance of loading
        all items or checking if any current formulae/casks have Ruby issues.
      EOS
      switch "--aliases",
             description: "Verify any alias symlinks in each tap."
      switch "--syntax",
             description: "Syntax-check all of Homebrew's Ruby files (if no `<tap>` is passed)."

      named_args :tap
    end
  end

  def readall
    args = readall_args.parse

    if args.syntax? && args.no_named?
      scan_files = "#{HOMEBREW_LIBRARY_PATH}/**/*.rb"
      ruby_files = Dir.glob(scan_files).reject { |file| file =~ %r{/(vendor)/} }

      Homebrew.failed = true unless Readall.valid_ruby_syntax?(ruby_files)
    end

    options = { aliases: args.aliases? }
    taps = if args.no_named?
      Tap
    else
      args.named.to_installed_taps
    end
    taps.each do |tap|
      Homebrew.failed = true unless Readall.valid_tap?(tap, options)
    end
  end
end
