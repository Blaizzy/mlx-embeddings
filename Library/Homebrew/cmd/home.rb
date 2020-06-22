# frozen_string_literal: true

require "cli/parser"
require "cask/cask_loader"
require "cask/exceptions"

module Homebrew
  module_function

  def home_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `home` [<formula>]

        Open <formula>'s homepage in a browser, or open Homebrew's own homepage
        if no formula is provided.
      EOS
      switch :debug
    end
  end

  def home
    home_args.parse

    if args.no_named?
      exec_browser HOMEBREW_WWW
    else
      homepages = args.named.map do |ref|
        Formulary.factory(ref).homepage
      rescue FormulaUnavailableError => e
        formula_error_message = e.message
        begin
          cask = Cask::CaskLoader.load(ref)
          puts "Formula \"#{ref}\" not found. Found a cask instead."
          cask.homepage
        rescue Cask::CaskUnavailableError => e
          cask_error_message = e.message
          odie "No available formula or cask with the name \"#{name}\"\n" \
               "#{formula_error_message}\n" \
               "#{cask_error_message}\n"
        end
      end
      exec_browser(*homepages)
    end
  end
end
