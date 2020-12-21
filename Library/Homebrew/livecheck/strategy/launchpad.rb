# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Launchpad} strategy identifies versions of software at
      # launchpad.net by checking the main page for a project.
      #
      # Launchpad URLs take a variety of formats but all the current formats
      # contain the project name as the first part of the URL path:
      #
      # * `https://launchpad.net/example/1.2/1.2.3/+download/example-1.2.3.tar.gz`
      # * `https://launchpad.net/example/trunk/1.2.3/+download/example-1.2.3.tar.gz`
      # * `https://code.launchpad.net/example/1.2/1.2.3/+download/example-1.2.3.tar.gz`
      #
      # The default regex identifies the latest version within an HTML element
      # found on the main page for a project:
      #
      # <pre><div class="version">
      #   Latest version is 1.2.3
      # </div></pre>
      #
      # @api public
      class Launchpad
        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{
          ^https?://(?:[^/]+?\.)*launchpad\.net
          /(?<project_name>[^/]+) # The Launchpad project name
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

          # The main page for the project on Launchpad
          page_url = "https://launchpad.net/#{match[:project_name]}"

          # The default regex is the same for all URLs using this strategy
          regex ||= %r{class="[^"]*version[^"]*"[^>]*>\s*Latest version is (.+)\s*</}

          PageMatch.find_versions(page_url, regex, &block)
        end
      end
    end
  end
end
