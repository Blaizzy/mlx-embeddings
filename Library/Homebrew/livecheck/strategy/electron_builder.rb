# typed: true
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {ElectronBuilder} strategy fetches content at a URL and parses
      # it as an electron-builder appcast in YAML format.
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
        # @return [Array]
        sig {
          params(
            content: String,
            block:   T.nilable(
              T.proc.params(arg0: T::Hash[String, T.untyped]).returns(T.any(String, T::Array[String], NilClass)),
            ),
          ).returns(T::Array[String])
        }
        def self.versions_from_content(content, &block)
          require "yaml"

          yaml = YAML.safe_load(content)
          return [] if yaml.blank?

          return Strategy.handle_block_return(block.call(yaml)) if block

          version = yaml["version"]
          version.present? ? [version] : []
        end

        # Checks the YAML content at the URL for new versions.
        #
        # @param url [String] the URL of the content to check
        # @return [Hash]
        sig {
          params(
            url:    String,
            unused: T.nilable(T::Hash[Symbol, T.untyped]),
            block:  T.nilable(T.proc.params(arg0: T::Hash[String, T.untyped]).returns(T.nilable(String))),
          ).returns(T::Hash[Symbol, T.untyped])
        }
        def self.find_versions(url:, **unused, &block)
          raise ArgumentError, "The #{T.must(name).demodulize} strategy does not support a regex." if unused[:regex]

          match_data = { matches: {}, url: url }

          match_data.merge!(Strategy.page_content(url))
          content = match_data.delete(:content)

          versions_from_content(content, &block).each do |version_text|
            match_data[:matches][version_text] = Version.new(version_text)
          end

          match_data
        end
      end
    end
  end
end
