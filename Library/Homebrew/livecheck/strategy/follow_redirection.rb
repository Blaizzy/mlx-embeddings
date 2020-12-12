# typed: false
# frozen_string_literal: true

require_relative "page_match"

module Homebrew
  module Livecheck
    module Strategy
      # The {FollowRedirection} strategy follows all URL redirections and scans
      # the final URL for matching text using the provided regex.
      #
      # @api private
      class FollowRedirection
        extend T::Sig

        NICE_NAME = "Follow HTTP Redirection"

        # We set the priority to zero since this cannot
        # be detected automatically.
        PRIORITY = 0

        # Whether the strategy can be applied to the provided URL.
        # FollowRedirection will technically match any HTTP URL but is
        # only usable with a `livecheck` block containing a regex.
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          url.match?(%r{^https?://})
        end

        # Checks the final URL for new versions after following all redirections,
        # using the provided regex for matching.
        sig { params(url: String, regex: T.nilable(Regexp)).returns(T::Hash[Symbol, T.untyped]) }
        def self.find_versions(url, regex)
          raise ArgumentError, "A regular expression is required for the #{NICE_NAME} strategy." if regex.nil?

          match_data = { matches: {}, regex: regex, url: url }

          if (location = Strategy.page_headers(url)["location"]) && (match = location[regex, 1])
            match_data[:matches][match] = Version.new(match)
          end

          match_data
        end
      end
    end
  end
end
