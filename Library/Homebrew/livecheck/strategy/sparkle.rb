# typed: true
# frozen_string_literal: true

require "bundle_version"

module Homebrew
  module Livecheck
    module Strategy
      # The {Sparkle} strategy fetches content at a URL and parses
      # it as a Sparkle appcast in XML format.
      #
      # This strategy is not applied automatically and it's necessary to use
      # `strategy :sparkle` in a `livecheck` block to apply it.
      #
      # @api private
      class Sparkle
        extend T::Sig

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
          # @api private
          :pub_date,
          # @api public
          :url,
          # @api private
          :bundle_version,
          keyword_init: true,
        ) do
          extend T::Sig

          extend Forwardable

          # @api public
          delegate version: :bundle_version

          # @api public
          delegate short_version: :bundle_version
        end

        # Identify version information from a Sparkle appcast.
        #
        # @param content [String] the text of the Sparkle appcast
        # @return [Item, nil]
        sig { params(content: String).returns(T.nilable(Item)) }
        def self.item_from_content(content)
          require "rexml/document"

          parsing_tries = 0
          xml = begin
            REXML::Document.new(content)
          rescue REXML::UndefinedNamespaceException => e
            undefined_prefix = e.to_s[/Undefined prefix ([^ ]+) found/i, 1]
            raise if undefined_prefix.blank?

            # Only retry parsing once after removing prefix from content
            parsing_tries += 1
            raise if parsing_tries > 1

            # When an XML document contains a prefix without a corresponding
            # namespace, it's necessary to remove the the prefix from the
            # content to be able to successfully parse it using REXML
            content = content.gsub(%r{(</?| )#{Regexp.escape(undefined_prefix)}:}, '\1')
            retry
          end

          # Remove prefixes, so we can reliably identify elements and attributes
          xml.root&.each_recursive do |node|
            node.prefix = ""
            node.attributes.each_attribute do |attribute|
              attribute.prefix = ""
            end
          end

          items = xml.get_elements("//rss//channel//item").map do |item|
            enclosure = item.elements["enclosure"]

            if enclosure
              url = enclosure["url"]
              short_version = enclosure["shortVersionString"]
              version = enclosure["version"]
              os = enclosure["os"]
            end

            url ||= item.elements["link"]&.text
            short_version ||= item.elements["shortVersionString"]&.text&.strip
            version ||= item.elements["version"]&.text&.strip

            title = item.elements["title"]&.text&.strip
            pub_date = item.elements["pubDate"]&.text&.strip&.presence&.yield_self do |date_string|
              Time.parse(date_string)
            rescue ArgumentError
              # Omit unparseable strings (e.g. non-English dates)
              nil
            end

            if (match = title&.match(/(\d+(?:\.\d+)*)\s*(\([^)]+\))?\Z/))
              short_version ||= match[1]
              version ||= match[2]
            end

            bundle_version = BundleVersion.new(short_version, version) if short_version || version

            next if os && os != "osx"

            if (minimum_system_version = item.elements["minimumSystemVersion"]&.text&.gsub(/\A\D+|\D+\z/, ""))
              macos_minimum_system_version = begin
                OS::Mac::Version.new(minimum_system_version).strip_patch
              rescue MacOSVersionError
                nil
              end

              next if macos_minimum_system_version&.prerelease?
            end

            data = {
              title:          title,
              pub_date:       pub_date || Time.new(0),
              url:            url,
              bundle_version: bundle_version,
            }.compact

            Item.new(**data) unless data.empty?
          end.compact

          items.max_by { |item| [item.pub_date, item.bundle_version] }
        end

        # Identify versions from content
        #
        # @param content [String] the content to pull version information from
        # @return [Array]
        sig {
          params(
            content: String,
            block:   T.nilable(T.proc.params(arg0: Item).returns(T.any(String, T::Array[String], NilClass))),
          ).returns(T::Array[String])
        }
        def self.versions_from_content(content, &block)
          item = item_from_content(content)
          return [] if item.blank?

          return Strategy.handle_block_return(block.call(item)) if block

          version = item.bundle_version&.nice_version
          version.present? ? [version] : []
        end

        # Checks the content at the URL for new versions.
        sig {
          params(
            url:    String,
            unused: T.nilable(T::Hash[Symbol, T.untyped]),
            block:  T.nilable(T.proc.params(arg0: Item).returns(T.nilable(String))),
          ).returns(T::Hash[Symbol, T.untyped])
        }
        def self.find_versions(url:, **unused, &block)
          raise ArgumentError, "The #{T.must(name).demodulize} strategy does not support a regex." if unused[:regex]

          match_data = { matches: {}, url: url }

          match_data.merge!(Strategy.page_content(url))
          content = match_data.delete(:content)

          versions_from_content(content, &block).each do |version_text|
            match_data[:matches][version_text] = Version.new(version_text)
          end

          match_data
        end
      end
    end
  end
end
