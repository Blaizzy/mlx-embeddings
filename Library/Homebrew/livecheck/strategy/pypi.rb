# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Pypi} strategy identifies versions of software at pypi.org by
      # checking project pages for archive files.
      #
      # PyPI URLs have a standard format but the hexadecimal text between
      # `/packages/` and the filename varies:
      #
      # * `https://files.pythonhosted.org/packages/<hex>/<hex>/<long_hex>/example-1.2.3.tar.gz`
      #
      # As such, the default regex only targets the filename at the end of the
      # URL.
      #
      # @api public
      class Pypi
        NICE_NAME = "PyPI"

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = /files\.pythonhosted\.org/i.freeze

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
          /
            (?<package_name>.+)- # The package name followed by a hyphen
            .*? # The version string
            (?<suffix>\.tar\.[a-z0-9]+|\.[a-z0-9]+)$ # Filename extension
          /ix =~ File.basename(url)

          # Use `\.t` instead of specific tarball extensions (e.g. .tar.gz)
          suffix.sub!(/\.t(?:ar\..+|[a-z0-9]+)$/i, "\.t")

          # It's not technically necessary to have the `#files` fragment at the
          # end of the URL but it makes the debug output a bit more useful.
          page_url = "https://pypi.org/project/#{package_name.gsub(/%20|_/, "-")}/#files"

          # Example regex: `%r{href=.*?/packages.*?/example[._-]v?(\d+(?:\.\d+)*).t}i`.
          regex ||=
            %r{href=.*?/packages.*?/#{Regexp.escape(package_name)}[._-]
               v?(\d+(?:\.\d+)*(.post\d+)?)#{Regexp.escape(suffix)}}ix

          PageMatch.find_versions(page_url, regex)
        end
      end
    end
  end
end
