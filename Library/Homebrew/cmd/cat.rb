# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def cat_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `cat` <formula>

        Display the source of <formula>.
      EOS
      max_named 1
    end
  end

  def cat
    cat_args.parse
    # do not "fix" this to support multiple arguments, the output would be
    # unparsable; if the user wants to cat multiple formula they can call
    # `brew cat` multiple times.
    formulae = Homebrew.args.formulae
    raise FormulaUnspecifiedError if formulae.empty?

    cd HOMEBREW_REPOSITORY
    pager = if ENV["HOMEBREW_BAT"].nil?
      "cat"
    else
      "#{HOMEBREW_PREFIX}/bin/bat"
    end
    safe_system pager, formulae.first.path, *Homebrew.args.passthrough
  end
end
