# typed: false
# frozen_string_literal: true

require "cask/cask"
require "cask/cask_loader"

module Homebrew
  module Search
    module Extension
      def search_descriptions(string_or_regex, args)
        super

        puts

        return if args.formula?

        ohai "Casks"
        Cask::Cask.all.extend(Searchable)
                  .search(string_or_regex, &:name)
                  .each do |cask|
          puts "#{Tty.bold}#{cask.token}:#{Tty.reset} #{cask.name.join(", ")}"
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

        cask_tokens = Tap.flat_map(&:cask_tokens)

        results = cask_tokens.extend(Searchable)
                             .search(string_or_regex)

        results |= DidYouMean::SpellChecker.new(dictionary: cask_tokens)
                                           .correct(string_or_regex)

        results.sort.map do |name|
          cask = Cask::CaskLoader.load(name)
          if cask.installed?
            pretty_installed(cask.token)
          else
            cask.token
          end
        end
      end
    end

    prepend Extension
  end
end
