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
      conflicts "--build-from-source", "--force-bottle"
    end
  end

  def __cache
    __cache_args.parse

    if args.no_named?
      puts HOMEBREW_CACHE
    else
      args.named.each do |name|
        formula = Formulary.factory name
        if Fetch.fetch_bottle?(formula)
          puts formula.bottle.cached_download
        else
          puts formula.cached_download
        end
      rescue FormulaUnavailableError => fe
        begin
          cask = Cask::CaskLoader.load name
          puts "cask: #{Cask::Cmd::Cache.cached_location(cask)}"
        rescue Cask::CaskUnavailableError => ce
          odie "No available formula or cask with the name \"#{name}\"\n" \
               "#{fe.message}\n" \
               "#{ce.message}\n"
        end
      end
    end
  end
end
