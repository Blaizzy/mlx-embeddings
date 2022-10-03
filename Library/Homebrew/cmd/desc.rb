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
        The cache is created on the first search, making that search slower than subsequent ones.
      EOS
      switch "-s", "--search",
             description: "Search both names and descriptions for <text>. If <text> is flanked by " \
                          "slashes, it is interpreted as a regular expression."
      switch "-n", "--name",
             description: "Search just names for <text>. If <text> is flanked by slashes, it is " \
                          "interpreted as a regular expression."
      switch "-d", "--description",
             description: "Search just descriptions for <text>. If <text> is flanked by slashes, " \
                          "it is interpreted as a regular expression."
      switch "--eval-all",
             description: "Evaluate all available formulae and casks, whether installed or not, to search their " \
                          "descriptions. Implied if HOMEBREW_EVAL_ALL is set."
      switch "--formula", "--formulae",
             description: "Treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "Treat all named arguments as casks."

      conflicts "--search", "--name", "--description"

      named_args [:formula, :cask, :text_or_regex], min: 1
    end
  end

  def desc
    args = desc_args.parse

    if !args.eval_all? && !Homebrew::EnvConfig.eval_all?
      odeprecated "brew desc", "brew desc --eval-all or HOMEBREW_EVAL_ALL"
    end

    search_type = if args.search?
      :either
    elsif args.name?
      :name
    elsif args.description?
      :desc
    end

    if search_type.blank?
      desc = {}
      args.named.to_formulae_and_casks.each do |formula_or_cask|
        if formula_or_cask.is_a? Formula
          desc[formula_or_cask.full_name] = formula_or_cask.desc
        else
          description = formula_or_cask.desc.presence || Formatter.warning("[no description]")
          desc[formula_or_cask.full_name] = "(#{formula_or_cask.name.join(", ")}) #{description}"
        end
      end
      Descriptions.new(desc).print
    else
      query = args.named.join(" ")
      string_or_regex = query_regexp(query)
      unless args.cask?
        ohai "Formulae"
        CacheStoreDatabase.use(:descriptions) do |db|
          cache_store = DescriptionCacheStore.new(db)
          Descriptions.search(string_or_regex, search_type, cache_store).print
        end
      end
      unless args.formula?
        puts unless args.cask?
        ohai "Casks"
        CacheStoreDatabase.use(:cask_descriptions) do |db|
          cache_store = CaskDescriptionCacheStore.new(db)
          Descriptions.search(string_or_regex, search_type, cache_store).print
        end
      end
    end
  end
end
