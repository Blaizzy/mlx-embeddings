# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def mirror_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Reupload the stable URL of a formula for use as a mirror.
      EOS
      hide_from_man_page!
    end
  end

  def mirror
    odisabled "`brew mirror` (Bintray was shut down on 1st May 2021)"
  end
end
