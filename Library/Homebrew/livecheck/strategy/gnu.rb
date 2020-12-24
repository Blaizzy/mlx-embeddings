# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Gnu} strategy identifies versions of software at gnu.org by
      # checking directory listing pages.
      #
      # GNU URLs use a variety of formats:
      #
      # * Archive file URLs:
      #   * `https://ftp.gnu.org/gnu/example/example-1.2.3.tar.gz`
      #   * `https://ftp.gnu.org/gnu/example/1.2.3/example-1.2.3.tar.gz`
      # * Homepage URLs:
      #   * `https://www.gnu.org/software/example/`
      #   * `https://example.gnu.org`
      #
      # There are other URL formats that this strategy currently doesn't
      # support:
      #
      # * `https://ftp.gnu.org/non-gnu/example/source/feature/1.2.3/example-1.2.3.tar.gz`
      # * `https://savannah.nongnu.org/download/example/example-1.2.3.tar.gz`
      # * `https://download.savannah.gnu.org/releases/example/example-1.2.3.tar.gz`
      # * `https://download.savannah.nongnu.org/releases/example/example-1.2.3.tar.gz`
      #
      # The default regex identifies versions in archive files found in `href`
      # attributes.
      #
      # @api public
      class Gnu
        NICE_NAME = "GNU"

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{
          //.+?\.gnu\.org$|
          gnu\.org/(?:gnu|software)/
        }ix.freeze

        # The `Regexp` used to parse the project name from the provided URL.
        # The strategy uses this information to create the URL to check and
        # the default regex.
        PROJECT_NAME_REGEXES = [
          %r{/(?:gnu|software)/(?<project_name>.+?)/}i,
          %r{//(?<project_name>.+?)\.gnu\.org(?:/)?$}i,
        ].freeze

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        def self.match?(url)
          URL_MATCH_REGEX.match?(url) && url.exclude?("savannah.")
        end

        # Generates a URL and regex (if one isn't provided) and passes them
        # to {PageMatch.find_versions} to identify versions in the content.
        #
        # @param url [String] the URL of the content to check
        # @param regex [Regexp] a regex used for matching versions in content
        # @return [Hash]
        def self.find_versions(url, regex = nil)
          project_names = PROJECT_NAME_REGEXES.map do |project_name_regex|
            m = url.match(project_name_regex)
            m["project_name"] if m
          end.compact
          return { matches: {}, regex: regex, url: url } if project_names.blank?

          if project_names.length > 1
            odebug <<~EOS

              Multiple project names found: #{match_list}

            EOS
          end

          project_name = project_names.first

          # The directory listing page for the project's files
          page_url = "http://ftp.gnu.org/gnu/#{project_name}/?C=M&O=D"

          # The default regex consists of the following parts:
          # * `href=.*?`: restricts matching to URLs in `href` attributes
          # * The project name
          # * `[._-]`: the generic delimiter between project name and version
          # * `v?(\d+(?:\.\d+)*)`: the numeric version
          # * `(?:\.[a-z]+|/)`: the file extension (a trailing delimiter)
          #
          # Example regex: `%r{href=.*?example[._-]v?(\d+(?:\.\d+)*)(?:\.[a-z]+|/)}i`
          regex ||= %r{href=.*?#{project_name}[._-]v?(\d+(?:\.\d+)*)(?:\.[a-z]+|/)}i

          PageMatch.find_versions(page_url, regex)
        end
      end
    end
  end
end
