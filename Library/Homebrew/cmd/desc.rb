# typed: false
# frozen_string_literal: true

require "descriptions"
require "search"
require "description_cache_store"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  extend Search

  sig { returns(CLI::Parser) }
  def desc_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display <formula>'s name and one-line description.
        Formula descriptions are cached; the cache is created on the
        first search, making that search slower than subsequent ones.
      EOS
      switch "-s", "--search",
             description: "Search both names and descriptions for <text>. If <text> is flanked by "\
                          "slashes, it is interpreted as a regular expression."
      switch "-n", "--name",
             description: "Search just names for <text>. If <text> is flanked by slashes, it is "\
                          "interpreted as a regular expression."
      switch "-d", "--description",
             description: "Search just descriptions for <text>. If <text> is flanked by slashes, "\
                          "it is interpreted as a regular expression."

      conflicts "--search", "--name", "--description"

      named_args [:formula, :text_or_regex], min: 1
    end
  end

  def desc
    args = desc_args.parse

    search_type = if args.search?
      :either
    elsif args.name?
      :name
    elsif args.description?
      :desc
    end

    results = if search_type.nil?
      desc = {}
      args.named.to_formulae.each { |f| desc[f.full_name] = f.desc }
      Descriptions.new(desc)
    else
      query = args.named.join(" ")
      string_or_regex = query_regexp(query)
      CacheStoreDatabase.use(:descriptions) do |db|
        cache_store = DescriptionCacheStore.new(db)
        Descriptions.search(string_or_regex, search_type, cache_store)
      end
    end

    results.print
  end
end
