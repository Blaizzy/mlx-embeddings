# typed: true
# frozen_string_literal: true

require "bundle_version"

module Homebrew
  module Livecheck
    module Strategy
      # The {Sparkle} strategy fetches content at a URL and parses it as a
      # Sparkle appcast in XML format.
      #
      # This strategy is not applied automatically and it's necessary to use
      # `strategy :sparkle` in a `livecheck` block to apply it.
      #
      # @api private
      class Sparkle
        # A priority of zero causes livecheck to skip the strategy. We do this
        # for {Sparkle} so we can selectively apply it when appropriate.
        PRIORITY = 0

        # The `Regexp` used to determine if the strategy applies to the URL.
        URL_MATCH_REGEX = %r{^https?://}i.freeze

        # Whether the strategy can be applied to the provided URL.
        #
        # @param url [String] the URL to match against
        # @return [Boolean]
        sig { params(url: String).returns(T::Boolean) }
        def self.match?(url)
          URL_MATCH_REGEX.match?(url)
        end

        # @api private
        Item = Struct.new(
          # @api public
          :title,
          # @api public
          :channel,
          # @api private
          :pub_date,
          # @api public
          :url,
          # @api private
          :bundle_version,
          keyword_init: true,
        ) do
          extend Forwardable

          # @api public
          delegate version: :bundle_version

          # @api public
          delegate short_version: :bundle_version

          # @api public
          delegate nice_version: :bundle_version
        end

        # Identifies version information from a Sparkle appcast.
        #
        # @param content [String] the text of the Sparkle appcast
        # @return [Item, nil]
        sig { params(content: String).returns(T::Array[Item]) }
        def self.items_from_content(content)
          xml = Xml.parse_xml(content)
          return [] if xml.blank?

          # Remove prefixes, so we can reliably identify elements and attributes
          xml.root&.each_recursive do |node|
            node.prefix = ""
            node.attributes.each_attribute do |attribute|
              attribute.prefix = ""
            end
          end

          xml.get_elements("//rss//channel//item").map do |item|
            enclosure = item.elements["enclosure"]

            if enclosure
              url = enclosure["url"]
              short_version = enclosure["shortVersionString"]
              version = enclosure["version"]
              os = enclosure["os"]
            end

            channel = item.elements["channel"]&.text
            url ||= item.elements["link"]&.text
            short_version ||= item.elements["shortVersionString"]&.text&.strip
            version ||= item.elements["version"]&.text&.strip

            title = item.elements["title"]&.text&.strip
            pub_date = item.elements["pubDate"]&.text&.strip&.presence&.then do |date_string|
              Time.parse(date_string)
            rescue ArgumentError
              # Omit unparsable strings (e.g. non-English dates)
              nil
            end

            if (match = title&.match(/(\d+(?:\.\d+)*)\s*(\([^)]+\))?\Z/))
              short_version ||= match[1]
              version ||= match[2]
            end

            bundle_version = BundleVersion.new(short_version, version) if short_version || version

            next if os && !((os == "osx") || (os == "macos"))

            if (minimum_system_version = item.elements["minimumSystemVersion"]&.text&.gsub(/\A\D+|\D+\z/, ""))
              macos_minimum_system_version = begin
                MacOSVersion.new(minimum_system_version).strip_patch
              rescue MacOSVersion::Error
                nil
              end

              next if macos_minimum_system_version&.prerelease?
            end

            data = {
              title:          title,
              channel:        channel,
              pub_date:       pub_date,
              url:            url,
              bundle_version: bundle_version,
            }.compact
            next if data.empty?

            # Set a default `pub_date` (for sorting) if one isn't provided
            data[:pub_date] ||= Time.new(0)

            Item.new(**data)
          end.compact
        end

        # Uses `#items_from_content` to identify versions from the Sparkle
        # appcast content or, if a block is provided, passes the content to
        # the block to handle matching.
        #
        # @param content [String] the content to check
        # @param regex [Regexp, nil] a regex for use in a strategy block
        # @return [Array]
        sig {
          params(
            content: String,
            regex:   T.nilable(Regexp),
            block:   T.nilable(Proc),
          ).returns(T::Array[String])
        }
        def self.versions_from_content(content, regex = nil, &block)
          items = items_from_content(content).sort_by { |item| [item.pub_date, item.bundle_version] }.reverse
          return [] if items.blank?

          item = items.first

          if block
            block_return_value = case block.parameters[0]
            when [:opt, :item], [:rest], [:req]
              regex.present? ? yield(item, regex) : yield(item)
            when [:opt, :items]
              regex.present? ? yield(items, regex) : yield(items)
            else
              raise "First argument of Sparkle `strategy` block must be `item` or `items`"
            end
            return Strategy.handle_block_return(block_return_value)
          end

          version = T.must(item).bundle_version&.nice_version
          version.present? ? [version] : []
        end

        # Checks the content at the URL for new versions.
        #
        # @param url [String] the URL of the content to check
        # @param regex [Regexp, nil] a regex for use in a strategy block
        # @return [Hash]
        sig {
          params(
            url:     String,
            regex:   T.nilable(Regexp),
            _unused: T.nilable(T::Hash[Symbol, T.untyped]),
            block:   T.nilable(Proc),
          ).returns(T::Hash[Symbol, T.untyped])
        }
        def self.find_versions(url:, regex: nil, **_unused, &block)
          if regex.present? && block.blank?
            raise ArgumentError,
                  "#{Utils.demodulize(T.must(name))} only supports a regex when using a `strategy` block"
          end

          match_data = { matches: {}, regex: regex, url: url }

          match_data.merge!(Strategy.page_content(url))
          content = match_data.delete(:content)
          return match_data if content.blank?

          versions_from_content(content, regex, &block).each do |version_text|
            match_data[:matches][version_text] = Version.new(version_text)
          end

          match_data
        end
      end
    end
  end
end
