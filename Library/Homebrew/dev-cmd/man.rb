# typed: false
# frozen_string_literal: true

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def man_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Generate Homebrew's manpages.
      EOS
      switch "--fail-if-not-changed",
             description: "Return a failing status code if no changes are detected in the manpage outputs. "\
                          "This can be used to notify CI when the manpages are out of date. Additionally, "\
                          "the date used in new manpages will match those in the existing manpages (to allow "\
                          "comparison without factoring in the date)."
      named_args :none

      hide_from_man_page!
    end
  end

  def man
    odisabled "`brew man`", "`brew generate-man-completions`"
  end
end
