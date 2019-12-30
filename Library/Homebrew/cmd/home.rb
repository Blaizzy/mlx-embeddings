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
      switch :debug
    end
  end

  def home
    home_args.parse

    if args.remaining.empty?
      exec_browser HOMEBREW_WWW
    else
      exec_browser(*Homebrew.args.formulae.map(&:homepage))
    end
  end
end
