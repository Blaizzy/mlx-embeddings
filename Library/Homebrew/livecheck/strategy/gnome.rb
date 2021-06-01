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
      # Before version 40, GNOME used a version scheme where unstable releases
      # were indicated with a minor that's 90+ or odd. The newer version scheme
      # uses trailing alpha/beta/rc text to identify unstable versions
      # (e.g., `40.alpha`).
      #
      # When a regex isn't provided in a `livecheck` block, the strategy uses
      # a default regex that matches versions which don't include trailing text
      # after the numeric version (e.g., `40.0` instead of `40.alpha`) and it
      # selectively filters out unstable versions below 40 using the rules for
      # the older version scheme.
      #
      # @api public
      class Gnome
        extend T::Sig

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
        sig {
          params(
            url:   String,
            regex: T.nilable(Regexp),
            cask:  T.nilable(Cask::Cask),
            block: T.nilable(T.proc.params(arg0: String).returns(T.any(T::Array[String], String))),
          ).returns(T::Hash[Symbol, T.untyped])
        }
        def self.find_versions(url, regex, cask: nil, &block)
          match = url.match(URL_MATCH_REGEX)

          page_url = "https://download.gnome.org/sources/#{match[:package_name]}/cache.json"

          if regex.blank?
            # GNOME archive files seem to use a standard filename format, so we
            # count on the delimiter between the package name and numeric
            # version being a hyphen and the file being a tarball.
            regex = /#{Regexp.escape(match[:package_name])}-(\d+(?:\.\d+)+)\.t/i
            version_data = PageMatch.find_versions(page_url, regex, cask: cask, &block)

            # Filter out unstable versions using the old version scheme where
            # the major version is below 40.
            version_data[:matches].reject! do |_, version|
              version.major < 40 && (version.minor >= 90 || version.minor.to_i.odd?)
            end

            version_data
          else
            PageMatch.find_versions(page_url, regex, cask: cask, &block)
          end
        end
      end
    end
  end
end
