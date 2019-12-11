# frozen_string_literal: true

require "fetch"
require "cli/parser"

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

    if ARGV.named.empty?
      puts HOMEBREW_CACHE
    else
      Homebrew.args.formulae.each do |f|
        if Fetch.fetch_bottle?(f)
          puts f.bottle.cached_download
        else
          puts f.cached_download
        end
      end
    end
  end
end
