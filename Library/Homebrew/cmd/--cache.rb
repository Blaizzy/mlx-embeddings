# frozen_string_literal: true

require "fetch"
require "cli/parser"
require "cask/download"

module Homebrew
  extend Fetch

  module_function

  def __cache_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--cache` [<options>] [<formula|cask>]

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
    args = __cache_args.parse

    if args.no_named?
      puts HOMEBREW_CACHE
      return
    end

    formulae_or_casks = if args.formula?
      args.named.to_formulae
    elsif args.cask?
      args.named.to_casks
    else
      args.named.to_formulae_and_casks
    end

    formulae_or_casks.each do |formula_or_cask|
      if formula_or_cask.is_a? Formula
        print_formula_cache formula_or_cask, args: args
      else
        print_cask_cache formula_or_cask
      end
    end
  end

  def print_formula_cache(formula, args:)
    if fetch_bottle?(formula, args: args)
      puts formula.bottle.cached_download
    else
      puts formula.cached_download
    end
  end

  def print_cask_cache(cask)
    puts Cask::Download.new(cask).downloader.cached_location
  end
end
