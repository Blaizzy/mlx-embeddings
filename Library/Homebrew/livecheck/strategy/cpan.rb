# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Cpan} strategy identifies versions of software at
      # cpan.metacpan.org by checking directory listing pages.
      #
      # CPAN URLs take the following formats:
      #
      # * `https://cpan.metacpan.org/authors/id/H/HO/HOMEBREW/Brew-v1.2.3.tar.gz`
      # * `https://cpan.metacpan.org/authors/id/H/HO/HOMEBREW/brew/brew-v1.2.3.tar.gz`
      #
      # In these examples, `HOMEBREW` is the author name and the preceding `H`
      # and `HO` directories correspond to the first letter(s). Some authors
      # also store files in subdirectories, as in the second example above.
      #
      # @api public
      class Cpan
        NICE_NAME = "CPAN"

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{
          ^https?://cpan\.metacpan\.org
          (?<path>/authors/id(?:/[^/]+){3,}/) # Path before the filename
          (?<prefix>[^/]+) # Filename text before the version
          -v?\d+(?:\.\d+)* # The numeric version
          (?<suffix>[^/]+) # Filename text after the version
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

          # The directory listing page where the archive files are found
          page_url = "https://cpan.metacpan.org#{match[:path]}"

          # Example regex: `/href=.*?Brew[._-]v?(\d+(?:\.\d+)*)\.t/i`
          regex ||= /href=.*?#{match[:prefix]}[._-]v?(\d+(?:\.\d+)*)#{Regexp.escape(suffix)}/i

          PageMatch.find_versions(page_url, regex, &block)
        end
      end
    end
  end
end
