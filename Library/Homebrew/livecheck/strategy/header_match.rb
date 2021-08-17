# typed: true
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {HeaderMatch} strategy follows all URL redirections and scans
      # the resulting headers for matching text using the provided regex.
      #
      # This strategy is not applied automatically and it's necessary to use
      # `strategy :header_match` in a `livecheck` block to apply it.
      #
      # @api private
      class HeaderMatch
        extend T::Sig

        NICE_NAME = "Header match"

        # A priority of zero causes livecheck to skip the strategy. We do this
        # for {HeaderMatch} so we can selectively apply it when appropriate.
        PRIORITY = 0

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{^https?://}i.freeze

        # The header fields to check when a `strategy` block isn't provided.
        DEFAULT_HEADERS_TO_CHECK = ["content-disposition", "location"].freeze

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          URL_MATCH_REGEX.match?(url)
        end

        # Identify versions from HTTP headers.
        #
        # @param headers [Hash] a hash of HTTP headers to check for versions
        # @param regex [Regexp, nil] a regex for matching versions
        # @return [Array]
        sig {
          params(
            headers: T::Hash[String, String],
            regex:   T.nilable(Regexp),
            block:   T.nilable(
              T.proc.params(
                arg0: T::Hash[String, String],
                arg1: T.nilable(Regexp),
              ).returns(T.any(String, T::Array[String], NilClass)),
            ),
          ).returns(T::Array[String])
        }
        def self.versions_from_headers(headers, regex = nil, &block)
          return Strategy.handle_block_return(block.call(headers, regex)) if block

          DEFAULT_HEADERS_TO_CHECK.map do |header_name|
            header_value = headers[header_name]
            next if header_value.blank?

            if regex
              header_value[regex, 1]
            else
              v = Version.parse(header_value, detected_from_url: true)
              v.null? ? nil : v.to_s
            end
          end.compact.uniq
        end

        # Checks the final URL for new versions after following all redirections,
        # using the provided regex for matching.
        #
        # @param url [String] the URL to fetch
        # @param regex [Regexp, nil] a regex used for matching versions
        # @return [Hash]
        sig {
          params(
            url:     String,
            regex:   T.nilable(Regexp),
            _unused: T.nilable(T::Hash[Symbol, T.untyped]),
            block:   T.nilable(
              T.proc.params(arg0: T::Hash[String, String], arg1: T.nilable(Regexp)).returns(T.nilable(String)),
            ),
          ).returns(T::Hash[Symbol, T.untyped])
        }
        def self.find_versions(url:, regex: nil, **_unused, &block)
          match_data = { matches: {}, regex: regex, url: url }

          headers = Strategy.page_headers(url)

          # Merge the headers from all responses into one hash
          merged_headers = headers.reduce(&:merge)
          return match_data if merged_headers.blank?

          versions_from_headers(merged_headers, regex, &block).each do |version_text|
            match_data[:matches][version_text] = Version.new(version_text)
          end

          match_data
        end
      end
    end
  end
end
