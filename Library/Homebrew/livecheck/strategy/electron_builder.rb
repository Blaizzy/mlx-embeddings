# typed: true
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {ElectronBuilder} strategy fetches content at a URL and parses it
      # as an electron-builder appcast in YAML format.
      #
      # This strategy is not applied automatically and it's necessary to use
      # `strategy :electron_builder` in a `livecheck` block to apply it.
      #
      # @api private
      class ElectronBuilder
        extend T::Sig

        NICE_NAME = "electron-builder"

        # A priority of zero causes livecheck to skip the strategy. We do this
        # for {ElectronBuilder} so we can selectively apply it when appropriate.
        PRIORITY = 0

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{^https?://.+/[^/]+\.ya?ml(?:\?[^/?]+)?$}i.freeze

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          URL_MATCH_REGEX.match?(url)
        end

        # Parses YAML text and identifies versions in it.
        #
        # @param content [String] the YAML text to parse and check
        # @param regex [Regexp, nil] a regex for use in a strategy block
        # @return [Array]
        sig {
          params(
            content: String,
            regex:   T.nilable(Regexp),
            block:   T.untyped,
          ).returns(T::Array[String])
        }
        def self.versions_from_content(content, regex = nil, &block)
          require "yaml"

          yaml = YAML.safe_load(content)
          return [] if yaml.blank?

          if block
            block_return_value = regex.present? ? yield(yaml, regex) : yield(yaml)
            return Strategy.handle_block_return(block_return_value)
          end

          version = yaml["version"]
          version.present? ? [version] : []
        end

        # Checks the YAML content at the URL for new versions.
        #
        # @param url [String] the URL of the content to check
        # @return [Hash]
        sig {
          params(
            url:     String,
            regex:   T.nilable(Regexp),
            _unused: T.nilable(T::Hash[Symbol, T.untyped]),
            block:   T.untyped,
          ).returns(T::Hash[Symbol, T.untyped])
        }
        def self.find_versions(url:, regex: nil, **_unused, &block)
          if regex.present? && block.blank?
            raise ArgumentError, "#{T.must(name).demodulize} only supports a regex when using a `strategy` block"
          end

          match_data = { matches: {}, regex: regex, url: url }

          match_data.merge!(Strategy.page_content(url))
          content = match_data.delete(:content)

          versions_from_content(content, regex, &block).each do |version_text|
            match_data[:matches][version_text] = Version.new(version_text)
          end

          match_data
        end
      end
    end
  end
end
