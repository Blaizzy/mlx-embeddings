# typed: false
# frozen_string_literal: true

require "cask/cask"
require "cask/cask_loader"

module Homebrew
  module Search
    module Extension
      def search_descriptions(string_or_regex, args)
        super

        return if args.formula?

        puts unless args.cask?
        ohai "Casks"
        CacheStoreDatabase.use(:cask_descriptions) do |db|
          cache_store = CaskDescriptionCacheStore.new(db)
          Descriptions.search(string_or_regex, :desc, cache_store).print
        end
      end

      def search_casks(string_or_regex)
        if string_or_regex.is_a?(String) && string_or_regex.match?(HOMEBREW_TAP_CASK_REGEX)
          return begin
            [Cask::CaskLoader.load(string_or_regex).token]
          rescue Cask::CaskUnavailableError
            []
          end
        end

        cask_tokens = Tap.flat_map(&:cask_tokens).map do |c|
          c.sub(%r{^homebrew/cask.*/}, "")
        end

        results = cask_tokens.extend(Searchable)
                             .search(string_or_regex)

        cask_names = Cask::Cask.all.map(&:full_name)
        results += DidYouMean::SpellChecker.new(dictionary: cask_names)
                                           .correct(string_or_regex)

        results.sort.map do |name|
          cask = Cask::CaskLoader.load(name)
          if cask.installed?
            pretty_installed(cask.full_name)
          else
            cask.full_name
          end
        end.uniq
      end
    end

    prepend Extension
  end
end
