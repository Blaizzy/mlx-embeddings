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
    end
  end

  def home
    args = home_args.parse

    if args.no_named?
      exec_browser HOMEBREW_WWW
    else
      homepages = args.named.map do |name|
        f = Formulary.factory(name)
        puts "Opening homepage for formula #{name}"
        f.homepage
      rescue FormulaUnavailableError
        begin
          c = Cask::CaskLoader.load(name)
          puts "Opening homepage for cask #{name}"
          c.homepage
        rescue Cask::CaskUnavailableError
          odie "No available formula or cask with the name \"#{name}\""
        end
      end
      exec_browser(*homepages)
    end
  end
end
