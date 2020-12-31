# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Gnome} strategy identifies versions of software at gnome.org by
      # checking the available downloads found in a project's `cache.json`
      # file.
      #
      # GNOME URLs generally follow a standard format:
      #
      # * `https://download.gnome.org/sources/example/1.2/example-1.2.3.tar.xz`
      #
      # The default regex restricts matching to filenames containing a version
      # with an even-numbered minor below 90, as these are stable releases.
      #
      # @api public
      class Gnome
        NICE_NAME = "GNOME"

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{
          ^https?://download\.gnome\.org
          /sources
          /(?<package_name>[^/]+)/ # The GNOME package name
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

          page_url = "https://download.gnome.org/sources/#{match[:package_name]}/cache.json"

          # GNOME archive files seem to use a standard filename format, so we
          # count on the delimiter between the package name and numeric version
          # being a hyphen and the file being a tarball.
          #
          # The `([0-8]\d*?)?[02468]` part of the regex is intended to restrict
          # matching to versions with an even-numbered minor, as these are
          # stable releases. This also excludes x.90+ versions, which are
          # development versions. See: https://www.gnome.org/gnome-3/source/
          #
          # Example regex: `/example-(\d+\.([0-8]\d*?)?[02468](?:\.\d+)*?)\.t/i`
          regex ||= /#{Regexp.escape(match[:package_name])}-(\d+\.([0-8]\d*?)?[02468](?:\.\d+)*?)\.t/i

          PageMatch.find_versions(page_url, regex, &block)
        end
      end
    end
  end
end
