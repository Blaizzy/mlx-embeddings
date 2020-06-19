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
      homepages = args.named.flat_map do |name|
        begin
          [Formulary.factory(name).homepage]
        rescue FormulaUnavailableError => e
          puts e.message
          begin
            cask = Cask::CaskLoader.load(name)
            puts "Found a cask named \"#{name}\" instead."
            [cask.homepage]
          rescue Cask::CaskUnavailableError
            []
          end
        end
      end
      exec_browser *homepages
    end
  end
end
