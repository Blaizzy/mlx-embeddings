# typed: false
# frozen_string_literal: true

require "bundle_version"
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
          return false unless url.match?(%r{^https?://})

          xml = url.end_with?(".xml")
          xml ||= begin
            headers = Strategy.page_headers(url)
            content_type = headers["content-type"]&.split(";", 2)&.first
            ["application/xml", "text/xml"].include?(content_type)
          end
          return false unless xml

          contents = Strategy.page_contents(url)
          contents.include?("http://www.andymatuschak.org/xml-namespaces/sparkle")
        end

        # Checks the content at the URL for new versions.
        sig { params(url: String, regex: T.nilable(Regexp)).returns(T::Hash[Symbol, T.untyped]) }
        def self.find_versions(url, regex, &block)
          raise ArgumentError, "The #{NICE_NAME} strategy does not support regular expressions." if regex

          require "nokogiri"

          match_data = { matches: {}, regex: regex, url: url }

          contents = Strategy.page_contents(url)

          xml = Nokogiri.parse(contents)
          xml.remove_namespaces!

          enclosure =
            xml.xpath("//rss//channel//item//enclosure")
               .map { |e| { url: e["url"], version: BundleVersion.new(e["shortVersionString"], e["version"]) } }
               .max_by { |e| e[:version] }

          if enclosure
            match = if block
              enclosure[:version] = enclosure[:version].nice_version
              block.call(enclosure).to_s
            else
              enclosure[:version].nice_version
            end

            match_data[:matches][match] = Version.new(match)
          end

          match_data
        end
      end
    end
  end
end
