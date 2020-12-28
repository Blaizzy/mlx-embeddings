# typed: false
# frozen_string_literal: true

require_relative "page_match"

module Homebrew
  module Livecheck
    module Strategy
      # The {HeaderMatch} strategy follows all URL redirections and scans
      # the resulting headers for matching text using the provided regex.
      #
      # @api private
      class HeaderMatch
        extend T::Sig

        NICE_NAME = "Header match"

        # A priority of zero causes livecheck to skip the strategy. We only
        # apply {HeaderMatch} using `strategy :header_match` in a `livecheck`
        # block, as we can't automatically determine when this can be
        # successfully applied to a URL.
        PRIORITY = 0

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{^https?://}i.freeze

        # Whether the strategy can be applied to the provided URL.
        # The strategy will technically match any HTTP URL but is
        # only usable with a `livecheck` block containing a regex
        # or block.
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          URL_MATCH_REGEX.match?(url)
        end

        # Checks the final URL for new versions after following all redirections,
        # using the provided regex for matching.
        sig { params(url: String, regex: T.nilable(Regexp)).returns(T::Hash[Symbol, T.untyped]) }
        def self.find_versions(url, regex, &block)
          match_data = { matches: {}, regex: regex, url: url }

          headers = Strategy.page_headers(url)

          # Merge the headers from all responses into one hash
          merged_headers = headers.reduce(&:merge)

          if block
            match = block.call(merged_headers, regex)
          else
            match = nil

            if (filename = merged_headers["content-disposition"])
              if regex
                match ||= filename[regex, 1]
              else
                v = Version.parse(filename, detected_from_url: true)
                match ||= v.to_s unless v.null?
              end
            end

            if (location = merged_headers["location"])
              if regex
                match ||= location[regex, 1]
              else
                v = Version.parse(location, detected_from_url: true)
                match ||= v.to_s unless v.null?
              end
            end
          end

          match_data[:matches][match] = Version.new(match) if match

          match_data
        end
      end
    end
  end
end
