# typed: false
# frozen_string_literal: true

require "open-uri"

module Homebrew
  module Livecheck
    module Strategy
      # The {PageMatch} strategy fetches content at a URL and scans it for
      # matching text using the provided regex.
      #
      # This strategy can be used in a `livecheck` block when no specific
      # strategies apply to a given URL. Though {PageMatch} will technically
      # match any HTTP URL, the strategy also requires a regex to function.
      #
      # The {find_versions} method is also used within other
      # strategies, to handle the process of identifying version text in
      # content.
      #
      # @api public
      class PageMatch
        NICE_NAME = "Page match"

        # A priority of zero causes livecheck to skip the strategy. We do this
        # for PageMatch so we can selectively apply the strategy only when a
        # regex is provided in a `livecheck` block.
        PRIORITY = 0

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{^https?://}i.freeze

        # Whether the strategy can be applied to the provided URL.
        # PageMatch will technically match any HTTP URL but is only
        # usable with a `livecheck` block containing a regex.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        def self.match?(url)
          URL_MATCH_REGEX.match?(url)
        end

        # Fetches the content at the URL, uses the regex to match text, and
        # returns an array of unique matches.
        #
        # @param url [String] the URL of the content to check
        # @param regex [Regexp] a regex used for matching versions in the
        #   content
        # @return [Array]
        def self.page_matches(url, regex, &block)
          page = Strategy.page_content(url)

          if block
            case (value = block.call(page))
            when String
              return [value]
            when Array
              return value
            else
              raise TypeError, "Return value of `strategy :page_match` block must be a string or array of strings."
            end
          end

          page.scan(regex).map do |match|
            case match
            when String
              match
            else
              match.first
            end
          end.uniq
        end

        # Checks the content at the URL for new versions, using the provided
        # regex for matching.
        #
        # @param url [String] the URL of the content to check
        # @param regex [Regexp] a regex used for matching versions in content
        # @return [Hash]
        def self.find_versions(url, regex, &block)
          match_data = { matches: {}, regex: regex, url: url }

          page_matches(url, regex, &block).each do |match|
            match_data[:matches][match] = Version.new(match)
          end

          match_data
        end
      end
    end
  end
end
