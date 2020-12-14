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

          return true if contents.match?(%r{https?://www.andymatuschak.org/xml-namespaces/sparkle})

          contents.include?("rss") &&
            contents.include?("channel") &&
            contents.include?("item") &&
            contents.include?("enclosure")
        end

        # Checks the content at the URL for new versions.
        sig { params(url: String, regex: T.nilable(Regexp)).returns(T::Hash[Symbol, T.untyped]) }
        def self.find_versions(url, regex, &block)
          raise ArgumentError, "The #{NICE_NAME} strategy does not support regular expressions." if regex

          require "nokogiri"

          match_data = { matches: {}, regex: regex, url: url }

          contents = Strategy.page_contents(url)

          xml = Nokogiri::XML(contents)
          xml.remove_namespaces!

          items = xml.xpath("//rss//channel//item").map do |item|
            enclosure = (item > "enclosure").first

            next unless enclosure

            short_version ||= enclosure["shortVersionString"]
            version ||= enclosure["version"]

            short_version ||= (item > "shortVersionString").first&.text
            version ||= (item > "version").first&.text

            data = {
              url:     enclosure["url"],
              version: short_version || version ? BundleVersion.new(short_version, version) : nil,
            }.compact

            data unless data.empty?
          end.compact

          item = items.max_by { |e| e[:version] }

          if item
            match = if block
              item[:version] = item[:version]&.nice_version
              block.call(item).to_s
            else
              item[:version]&.nice_version
            end

            match_data[:matches][match] = Version.new(match) if match
          end

          match_data
        end
      end
    end
  end
end
