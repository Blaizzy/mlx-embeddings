# frozen_string_literal: true

require "descriptions"
require "search"
require "description_cache_store"
require "cli/parser"

module Homebrew
  module_function

  extend Search

  def desc_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `desc` [<options>] (<text>|`/`<text>`/`|<formula>)

        Display <formula>'s name and one-line description.
        Formula descriptions are cached; the cache is created on the
        first search, making that search slower than subsequent ones.
      EOS
      flag   "-s", "--search=",
             description: "Search both names and descriptions for <text>. If <text> is flanked by "\
                          "slashes, it is interpreted as a regular expression."
      flag   "-n", "--name=",
             description: "Search just names for <text>. If <text> is flanked by slashes, it is "\
                          "interpreted as a regular expression."
      flag   "-d", "--description=",
             description: "Search just descriptions for <text>. If <text> is flanked by slashes, "\
                          "it is interpreted as a regular expression."
      switch :verbose
      conflicts "--search=", "--name=", "--description="
    end
  end

  def desc
    desc_args.parse

    search_type = []
    search_type << :either if args.search
    search_type << :name   if args.name
    search_type << :desc   if args.description
    odie "You must provide a search term." if search_type.present? && ARGV.named.empty?

    results = if search_type.empty?
      raise FormulaUnspecifiedError if ARGV.named.empty?

      desc = {}
      Homebrew.args.formulae.each { |f| desc[f.full_name] = f.desc }
      Descriptions.new(desc)
    else
      arg = ARGV.named.join(" ")
      string_or_regex = query_regexp(arg)
      CacheStoreDatabase.use(:descriptions) do |db|
        cache_store = DescriptionCacheStore.new(db)
        Descriptions.search(string_or_regex, search_type.first, cache_store)
      end
    end

    results.print
  end
end
