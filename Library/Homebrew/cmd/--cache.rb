# frozen_string_literal: true

require "fetch"
require "cli/parser"
require "cask/cmd"
require "cask/cask_loader"

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
      switch "--formula",
             description: "Only show cache files for formulae."
      switch "--cask",
             description: "Only show cache files for casks."
      conflicts "--build-from-source", "--force-bottle"
      conflicts "--formula", "--cask"
    end
  end

  def __cache
    __cache_args.parse

    if args.no_named?
      puts HOMEBREW_CACHE
    elsif args.formula?
      args.named.each do |name|
        print_formula_cache name
      end
    elsif args.cask?
      args.named.each do |name|
        print_cask_cache name
      end
    else
      args.named.each do |name|
        print_formula_cache name
      rescue FormulaUnavailableError
        begin
          print_cask_cache name
        rescue Cask::CaskUnavailableError
          odie "No available formula or cask with the name \"#{name}\""
        end
      end
    end
  end

  def print_formula_cache(name)
    formula = Formulary.factory name
    if Fetch.fetch_bottle?(formula)
      puts formula.bottle.cached_download
    else
      puts formula.cached_download
    end
  end

  def print_cask_cache(name)
    cask = Cask::CaskLoader.load name
    puts Cask::Cmd::Cache.cached_location(cask)
  end
end
