# typed: true
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
      class Sparkle < PageMatch
        extend T::Sig

        NICE_NAME = "Sparkle"

        # Checks the content at the URL for new versions.
        sig { params(url: String, regex: T.nilable(Regexp)).returns(T::Hash[Symbol, T.untyped]) }
        def self.find_versions(url, regex, &block)
          raise ArgumentError, "The #{name.demodulize} strategy does not support regular expressions." if regex

          match_data = { matches: {}, regex: regex, url: url }

          contents = Strategy.page_content(url)

          if (item = item_from_content(contents))
            match = if block
              block.call(item)&.to_s
            else
              item.bundle_version&.nice_version
            end

            match_data[:matches][match] = Version.new(match) if match
          end

          match_data
        end

        sig { params(content: String).returns(T.nilable(Item)) }
        def self.item_from_content(content)
          require "nokogiri"

          xml = Nokogiri::XML(content)
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

            bundle_version = BundleVersion.new(short_version, version) if short_version || version

            data = {
              title:          title,
              url:            url,
              bundle_version: bundle_version,
              short_version:  bundle_version&.short_version,
              version:        bundle_version&.version,
            }.compact

            Item.new(**data) unless data.empty?
          end.compact

          items.max_by(&:bundle_version)
        end
        private_class_method :item_from_content

        Item = Struct.new(:title, :url, :bundle_version, :short_version, :version, keyword_init: true) do
          extend T::Sig

          extend Forwardable

          delegate version: :bundle_version
          delegate short_version: :bundle_version
        end
      end
    end
  end
end
