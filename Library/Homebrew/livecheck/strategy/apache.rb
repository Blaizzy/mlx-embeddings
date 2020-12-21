# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Apache} strategy identifies versions of software at apache.org
      # by checking directory listing pages.
      #
      # Apache URLs start with `https://www.apache.org/dyn/closer.lua?path=`.
      # The `path` parameter takes one of the following formats:
      #
      # * `example/1.2.3/example-1.2.3.tar.gz`
      # * `example/example-1.2.3/example-1.2.3.tar.gz`
      # * `example/example-1.2.3-bin.tar.gz`
      #
      # When the `path` contains a version directory (e.g. `/1.2.3/`,
      # `/example-1.2.3/`, etc.), the default regex matches numeric versions
      # in directory names. Otherwise, the default regex matches numeric
      # versions in filenames.
      #
      # @api public
      class Apache
        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{
          ^https?://www\.apache\.org
          /dyn/.+path=
          (?<path>.+?)/      # Path to directory of files or version directories
          (?<prefix>[^/]*?)  # Any text in filename or directory before version
          v?\d+(?:\.\d+)+    # The numeric version
          (?<suffix>/|[^/]*) # Any text in filename or directory after version
        }ix.freeze

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
        def self.find_versions(url, regex = nil, &block)
          match = url.match(URL_MATCH_REGEX)

          # Use `\.t` instead of specific tarball extensions (e.g. .tar.gz)
          suffix = match[:suffix].sub(/\.t(?:ar\..+|[a-z0-9]+)$/i, "\.t")

          # Example URL: `https://archive.apache.org/dist/example/`
          page_url = "https://archive.apache.org/dist/#{match[:path]}/"

          # Example directory regex: `%r{href=["']?v?(\d+(?:\.\d+)+)/}i`
          # Example file regexes:
          # * `/href=["']?example-v?(\d+(?:\.\d+)+)\.t/i`
          # * `/href=["']?example-v?(\d+(?:\.\d+)+)-bin\.zip/i`
          regex ||= /href=["']?#{Regexp.escape(match[:prefix])}v?(\d+(?:\.\d+)+)#{Regexp.escape(suffix)}/i

          PageMatch.find_versions(page_url, regex, &block)
        end
      end
    end
  end
end
