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

            url = enclosure&.attr("url")
            short_version = enclosure&.attr("shortVersionString")
            version = enclosure&.attr("version")

            url ||= (item > "link").first&.text
            short_version ||= (item > "shortVersionString").first&.text&.strip
            version ||= (item > "version").first&.text&.strip

            title = (item > "title").first&.text&.strip

            if match = title&.match(/(\d+(?:\.\d+)*)\s*(\([^)]+\))?\Z/)
              short_version ||= match[1]
              version ||= match[2]
            end

            data = {
              title:          title,
              url:            url,
              bundle_version: short_version || version ? BundleVersion.new(short_version, version) : nil,
            }.compact

            data unless data.empty?
          end.compact

          item = items.max_by { |e| e[:bundle_version] }

          if item
            match = if block
              item[:short_version] = item[:bundle_version]&.short_version
              item[:version] = item[:bundle_version]&.version
              block.call(item).to_s
            else
              item[:bundle_version]&.nice_version
            end

            match_data[:matches][match] = Version.new(match) if match
          end

          match_data
        end
      end
    end
  end
end
