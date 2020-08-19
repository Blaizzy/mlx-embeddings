# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def home_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `home` [<formula>]

        Open <formula>'s homepage in a browser, or open Homebrew's own homepage
        if no formula is provided.
      EOS
    end
  end

  def home
    args = home_args.parse

    if args.no_named?
      exec_browser HOMEBREW_WWW
      return
    end

    homepages = args.named.to_formulae_and_casks.map do |formula_or_cask|
      puts "Opening homepage for #{name_of(formula_or_cask)}"
      formula_or_cask.homepage
    end

    exec_browser(*homepages)
  end

  def name_of(formula_or_cask)
    if formula_or_cask.is_a? Formula
      "Formula #{formula_or_cask.name}"
    else
      "Cask #{formula_or_cask.token}"
    end
  end
end
