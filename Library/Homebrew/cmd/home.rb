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

    homepages = args.formulae_and_casks.map do |formula_or_cask|
      disclaimer = disclaimers(formula_or_cask)
      disclaimer = " (#{disclaimer})" if disclaimer.present?

      puts "Opening homepage for #{name_of(formula_or_cask)}#{disclaimer}"
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

  def disclaimers(formula_or_cask)
    return unless formula_or_cask.is_a? Formula

    begin
      cask = Cask::CaskLoader.load formula_or_cask.name
      "for the cask, use #{cask.tap.name}/#{cask.token}"
    rescue Cask::CaskUnavailableError
      nil
    end
  end
end
