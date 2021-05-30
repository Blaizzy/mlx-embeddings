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
      # For the new GNOME versioning scheme used with GNOME 40 and newer, the
      # strategy matches all versions with a numeric minor (and micro if
      # present). This excludes versions like `40.alpha`, `40.beta` and `40.rc`
      # which are development releases.
      #
      # For the older GNOME versioning scheme used with major versions 3 and
      # below, only those filenames containing a version with an even-numbered
      # minor below 90 are matched, as these are stable releases.
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

            # Before GNOME 40, versions have a major equal to or less than 3.
            # Stable versions have an even-numbered minor less than 90.
            version_data[:matches].select! do |_, version|
              (version.major <= 3 && version.minor.to_i.even? && version.minor < 90) || (version.major >= 40)
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
