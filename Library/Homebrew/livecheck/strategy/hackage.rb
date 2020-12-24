# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Hackage} strategy identifies versions of software at
      # hackage.haskell.org by checking directory listing pages.
      #
      # Hackage URLs take one of the following formats:
      #
      # * `https://hackage.haskell.org/package/example-1.2.3/example-1.2.3.tar.gz`
      # * `https://downloads.haskell.org/~ghc/8.10.1/ghc-8.10.1-src.tar.xz`
      #
      # The default regex checks for the latest version in an `h3` heading element
      # with a format like `<h3>example-1.2.3/</h3>`.
      #
      # @api public
      class Hackage
        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = /(?:downloads|hackage)\.haskell\.org/i.freeze

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        def self.match?(url)
          URL_MATCH_REGEX.match?(url)
        end

        # Generates a URL and regex (if one isn't provided) and passes them
        # to {PageMatch.find_versions} to identify versions in the content.
        #
        # @param url [String] the URL of the content to check
        # @param regex [Regexp] a regex used for matching versions in content
        # @return [Hash]
        def self.find_versions(url, regex = nil)
          /^(?<package_name>.+?)-\d+/i =~ File.basename(url)

          # A page containing a directory listing of the latest source tarball
          page_url = "https://hackage.haskell.org/package/#{package_name}/src/"

          # Example regex: `%r{<h3>example-(.*?)/?</h3>}i`
          regex ||= %r{<h3>#{Regexp.escape(package_name)}-(.*?)/?</h3>}i

          PageMatch.find_versions(page_url, regex)
        end
      end
    end
  end
end
