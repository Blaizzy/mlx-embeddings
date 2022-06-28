# typed: false
# frozen_string_literal: true

require "fetch"
require "cli/parser"
require "cask/download"

module Homebrew
  extend T::Sig

  extend Fetch

  module_function

  sig { returns(CLI::Parser) }
  def __cache_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display Homebrew's download cache. See also `HOMEBREW_CACHE`.

        If <formula> is provided, display the file or directory used to cache <formula>.
      EOS
      switch "-s", "--build-from-source",
             description: "Show the cache file used when building from source."
      switch "--force-bottle",
             description: "Show the cache file used when pouring a bottle."
      flag "--bottle-tag=",
           description: "Show the cache file used when pouring a bottle for the given tag."
      switch "--HEAD",
             description: "Show the cache file used when building from HEAD."
      switch "--formula", "--formulae",
             description: "Only show cache files for formulae."
      switch "--cask", "--casks",
             description: "Only show cache files for casks."

      conflicts "--build-from-source", "--force-bottle", "--bottle-tag", "--HEAD", "--cask"
      conflicts "--formula", "--cask"

      named_args [:formula, :cask]
    end
  end

  sig { void }
  def __cache
    args = __cache_args.parse

    if args.no_named?
      puts HOMEBREW_CACHE
      return
    end

    formulae_or_casks = args.named.to_formulae_and_casks

    formulae_or_casks.each do |formula_or_cask|
      if formula_or_cask.is_a? Formula
        print_formula_cache formula_or_cask, args: args
      else
        print_cask_cache formula_or_cask
      end
    end
  end

  sig { params(formula: Formula, args: CLI::Args).void }
  def print_formula_cache(formula, args:)
    if fetch_bottle?(formula, args: args)
      puts formula.bottle_for_tag(args.bottle_tag&.to_sym).cached_download
    elsif args.HEAD?
      puts formula.head.cached_download
    else
      puts formula.cached_download
    end
  end

  sig { params(cask: Cask::Cask).void }
  def print_cask_cache(cask)
    puts Cask::Download.new(cask).downloader.cached_location
  end
end
