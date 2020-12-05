# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {GithubLatest} strategy identifies versions of software at
      # github.com by checking a repository's latest release page.
      #
      # GitHub URLs take a few different formats:
      #
      # * `https://github.com/example/example/releases/download/1.2.3/example-1.2.3.tar.gz`
      # * `https://github.com/example/example/archive/v1.2.3.tar.gz`
      # * `https://github.com/downloads/example/example/example-1.2.3.tar.gz`
      #
      # This strategy is used when latest releases are marked for software hosted
      # on GitHub. It is necessary to use `strategy :github_latest` in a `livecheck`
      # block for Livecheck to use this strategy.
      #
      # The default regex identifies versions from `href` attributes containing the
      # tag name.
      #
      # @api public
      class GithubLatest
        NICE_NAME = "GitHub - Latest"

        PRIORITY = 0

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{//github\.com(?:/downloads)?(?:/[^/]+){2}}i.freeze

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
          %r{github\.com/(?:downloads/)?(?<username>[^/]+)/(?<repository>[^/]+)}i =~ url.sub(/\.git$/i, "")

          # The page containing the latest release
          page_url = "https://github.com/#{username}/#{repository}/releases/latest"

          # The default regex applies to most repositories, but may have to be
          # replaced with a specific regex when the tag names contain the package
          # name or other characters apart from the version.
          regex ||= %r{href=.*?/tag/v?(\d+(?:\.\d+)+)["' >]}i

          Homebrew::Livecheck::Strategy::PageMatch.find_versions(page_url, regex)
        end
      end
    end
  end
end
