# typed: false
# frozen_string_literal: true

require "uri"

module Homebrew
  module Livecheck
    module Strategy
      # The {Cpan} strategy identifies versions of software at
      # cpan.metacpan.org by checking directory listing pages.
      #
      # CPAN URLs take the following format:
      #
      # * `https://cpan.metacpan.org/authors/id/M/MI/MIYAGAWA/Carton-v1.0.34.tar.gz`
      #
      # @api public
      class Cpan
        NICE_NAME = "CPAN"

        # The allowlist used to determine if the strategy applies to the URL.
        HOST_ALLOWLIST = ["cpan.metacpan.org"].freeze

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        def self.match?(url)
          uri = URI.parse url
          HOST_ALLOWLIST.include? uri.host
        rescue URI::InvalidURIError
          false
        end

        # Generates a URL and regex (if one isn't provided) and passes them
        # to {PageMatch.find_versions} to identify versions in the content.
        #
        # @param url [String] the URL of the content to check
        # @param regex [Regexp] a regex used for matching versions in content
        # @return [Hash]
        def self.find_versions(url, regex = nil)
          %r{
            /authors/id/(?<author_path>(?:[^/](?:/[^/]+){2})) # The author path (e.g. M/MI/MIYAGAWA)
            /(?<package_path>.+) # The package path (e.g. Carton-v1.0.34.tar.gz)
          }ix =~ url

          # We need a Pathname because we've monkeypatched extname to support
          # double extensions (e.g. tar.gz).
          pathname = Pathname.new(package_path)

          # Exteract package name
          /^(?<package_name>.+)-v?\d+/ =~ pathname.basename(pathname.extname)
          # Use `\.t` instead of specific tarball extensions (e.g. .tar.gz)
          suffix = pathname.extname.sub!(/\.t(?:ar\..+|[a-z0-9]+)$/i, "\.t")

          # A page containing a directory listing of the latest source tarball
          package_dir = Pathname.new(author_path).join(pathname.dirname)
          page_url = "https://cpan.metacpan.org/authors/id/#{package_dir}/"

          # Example regex: `%r{href=.*?Carton-v?(\d+(?:\.\d+)*).t}i`
          regex ||= /href=.*?#{package_name}-v?(\d+(?:\.\d+)*)#{Regexp.escape(suffix)}/i

          Homebrew::Livecheck::Strategy::PageMatch.find_versions(page_url, regex)
        end
      end
    end
  end
end
