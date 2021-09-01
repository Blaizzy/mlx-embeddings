# typed: true
# frozen_string_literal: true

module Utils
  module Bottles
    class << self
      undef tag

      def tag(symbol = nil)
        return Utils::Bottles::Tag.from_symbol(symbol) if symbol.present?

        Utils::Bottles::Tag.new(system: MacOS.version.to_sym, arch: Hardware::CPU.arch)
      end
    end

    class Collector
      private

      alias generic_find_matching_tag find_matching_tag

      def find_matching_tag(tag, no_older_versions: false)
        # Used primarily by developers testing beta macOS releases.
        if no_older_versions ||
           (OS::Mac.version.prerelease? &&
            Homebrew::EnvConfig.developer? &&
            Homebrew::EnvConfig.skip_or_later_bottles?)
          generic_find_matching_tag(tag)
        else
          generic_find_matching_tag(tag) ||
            find_older_compatible_tag(tag)
        end
      end

      # Find a bottle built for a previous version of macOS.
      def find_older_compatible_tag(tag)
        tag_version = begin
          tag.to_macos_version
        rescue MacOSVersionError
          nil
        end

        return if tag_version.blank?

        keys.find do |key|
          key_tag = Tag.from_symbol(key)
          next if key_tag.arch != tag.arch

          key_tag.to_macos_version <= tag_version
        rescue MacOSVersionError
          false
        end
      end
    end
  end
end
