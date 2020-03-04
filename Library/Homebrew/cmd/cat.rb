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
      named :formula
    end
  end

  def cat
    cat_args.parse

    cd HOMEBREW_REPOSITORY
    pager = if ENV["HOMEBREW_BAT"].nil?
      "cat"
    else
      "#{HOMEBREW_PREFIX}/bin/bat"
    end
    safe_system pager, args.formulae.first.path, *args.passthrough
  end
end
