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

        NICE_NAME = "Match HTTP Headers"

        # We set the priority to zero since this cannot
        # be detected automatically.
        PRIORITY = 0

        # Whether the strategy can be applied to the provided URL.
        # The strategy will technically match any HTTP URL but is
        # only usable with a `livecheck` block containing a regex
        # or block.
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          url.match?(%r{^https?://})
        end

        # Checks the final URL for new versions after following all redirections,
        # using the provided regex for matching.
        sig { params(url: String, regex: T.nilable(Regexp)).returns(T::Hash[Symbol, T.untyped]) }
        def self.find_versions(url, regex, &block)
          match_data = { matches: {}, regex: regex, url: url }

          data = { headers: Strategy.page_headers(url) }

          if (filename = data[:headers]["content-disposition"])
            if regex
              data[:version] ||= location[regex, 1]
            else
              v = Version.parse(filename, detected_from_url: true)
              data[:version] ||= v.to_s unless v.null?
            end
          end

          if (location = data[:headers]["location"])
            if regex
              data[:version] ||= location[regex, 1]
            else
              v = Version.parse(location, detected_from_url: true)
              data[:version] ||= v.to_s unless v.null?
            end
          end

          version = if block
            block.call(data)
          else
            data[:version]
          end

          match_data[:matches][version] = Version.new(version) if version

          match_data
        end
      end
    end
  end
end
