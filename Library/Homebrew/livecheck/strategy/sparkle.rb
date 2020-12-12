# typed: false
# frozen_string_literal: true

require_relative "page_match"

module Homebrew
  module Livecheck
    module Strategy
      # The {Sparkle} strategy fetches content at a URL and parses
      # its contents as a Sparkle appcast in XML format.
      #
      # @api private
      class Sparkle
        extend T::Sig

        NICE_NAME = "Sparkle"

        PRIORITY = 1

        # Whether the strategy can be applied to the provided URL.
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          url.match?(%r{^https?://}) &&
            ["application/xml", "text/xml"].include?(Strategy.page_headers(url)["content-type"]) &&
            Strategy.page_contents(url).include?("http://www.andymatuschak.org/xml-namespaces/sparkle")
        end

        # Checks the content at the URL for new versions.
        sig { params(url: String, regex: T.nilable(Regexp)).returns(T::Hash[Symbol, T.untyped]) }
        def self.find_versions(url, regex)
          raise ArgumentError, "The #{NICE_NAME} strategy does not support regular expressions." if regex

          require "nokogiri"

          match_data = { matches: {}, regex: regex, url: url }

          contents = Strategy.page_contents(url)

          xml = Nokogiri.parse(contents)
          xml.remove_namespaces!

          match = xml.xpath("//rss//channel//item//enclosure")
                     .map { |enclosure| [*enclosure["shortVersionString"], *enclosure["version"]].uniq }
                     .reject(&:empty?)
                     .uniq
                     .max_by { |versions| versions.map { |v| Version.new(v) } }
                     &.join(",")

          match_data[:matches][match] = Version.new(match) if match

          match_data
        end
      end
    end
  end
end
