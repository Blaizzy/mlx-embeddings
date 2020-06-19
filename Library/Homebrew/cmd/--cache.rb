# frozen_string_literal: true

require "fetch"
require "cli/parser"
require "cask/cmd"

module Homebrew
  module_function

  def __cache_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--cache` [<options>] [<formula>]

        Display Homebrew's download cache. See also `HOMEBREW_CACHE`.

        If <formula> is provided, display the file or directory used to cache <formula>.
      EOS
      switch "-s", "--build-from-source",
             description: "Show the cache file used when building from source."
      switch "--force-bottle",
             description: "Show the cache file used when pouring a bottle."
      conflicts "--build-from-source", "--force-bottle"
    end
  end

  def __cache
    __cache_args.parse

    if args.no_named?
      puts HOMEBREW_CACHE
    else
      args.formulae_and_casks.each do |formula_or_cask|
        case formula_or_cask
        when Formula
          formula = formula_or_cask
          if Fetch.fetch_bottle?(formula)
            puts formula.bottle.cached_download
          else
            puts formula.cached_download
          end
        when Cask::Cask
          cask = formula_or_cask
          puts "cask: #{Cask::Cmd::Cache.cached_location(cask)}"
        end
      end
    end
  end

end
