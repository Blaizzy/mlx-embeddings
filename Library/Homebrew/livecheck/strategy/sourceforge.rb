# typed: false
# frozen_string_literal: true

module Homebrew
  module Livecheck
    module Strategy
      # The {Sourceforge} strategy identifies versions of software at
      # sourceforge.net by checking a project's RSS feed.
      #
      # SourceForge URLs take a few different formats:
      #
      # * `https://downloads.sourceforge.net/project/example/example-1.2.3.tar.gz`
      # * `https://svn.code.sf.net/p/example/code/trunk`
      # * `:pserver:anonymous:@example.cvs.sourceforge.net:/cvsroot/example`
      #
      # The RSS feed for a project contains the most recent release archives
      # and while this is fine for most projects, this approach has some
      # shortcomings. Some project releases involve so many files that the one
      # we're interested in isn't present in the feed content. Some projects
      # contain additional software and the archive we're interested in is
      # pushed out of the feed (especially if it hasn't been updated recently).
      #
      # Usually we address this situation by adding a `livecheck` block to
      # the formula/cask that checks the page for the relevant directory in the
      # project instead. In this situation, it's necessary to use
      # `strategy :page_match` to prevent the {Sourceforge} stratgy from
      # being used.
      #
      # The default regex matches within `url` attributes in the RSS feed
      # and identifies versions within directory names or filenames.
      #
      # @api public
      class Sourceforge
        NICE_NAME = "SourceForge"

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{
          ^https?://(?:[^/]+?\.)*(?:sourceforge|sf)\.net
          (?:/projects?/(?<project_name>[^/]+)/
          |/p/(?<project_name>[^/]+)/
          |(?::/cvsroot)?/(?<project_name>[^/]+))
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

          page_url = "https://sourceforge.net/projects/#{match[:project_name]}/rss"

          # It may be possible to improve the default regex but there's quite a
          # bit of variation between projects and it can be challenging to
          # create something that works for most URLs.
          regex ||= %r{url=.*?/#{Regexp.escape(match[:project_name])}/files/.*?[-_/](\d+(?:[-.]\d+)+)[-_/%.]}i

          PageMatch.find_versions(page_url, regex, &block)
        end
      end
    end
  end
end
