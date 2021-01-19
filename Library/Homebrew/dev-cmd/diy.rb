# typed: true
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def diy_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Automatically determine the installation prefix for non-Homebrew software.
        Using the output from this command, you can install your own software into
        the Cellar and then link it into Homebrew's prefix with `brew link`.
      EOS
      flag   "--name=",
             description: "Explicitly set the <name> of the package being installed."
      flag   "--version=",
             description: "Explicitly set the <version> of the package being installed."

      max_named 0
      hide_from_man_page!
    end
  end

  def diy
    diy_args.parse

    odisabled "`brew diy`"
  end
end
