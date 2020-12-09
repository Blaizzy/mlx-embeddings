# typed: true
# frozen_string_literal: true

module Utils
  module Bottles
    class << self
      undef tag

      def tag
        if Hardware::CPU.intel?
          MacOS.version.to_sym
        else
          "#{Hardware::CPU.arch}_#{MacOS.version.to_sym}".to_sym
        end
      end
    end

    class Collector
      private

      alias generic_find_matching_tag find_matching_tag

      def find_matching_tag(tag)
        # Used primarily by developers testing beta macOS releases.
        if OS::Mac.prerelease? && Homebrew::EnvConfig.developer? &&
           Homebrew::EnvConfig.skip_or_later_bottles?
          generic_find_matching_tag(tag)
        else
          generic_find_matching_tag(tag) ||
            find_older_compatible_tag(tag)
        end
      end

      # Find a bottle built for a previous version of macOS.
      def find_older_compatible_tag(tag)
        tag_version = begin
          MacOS::Version.from_symbol(tag)
        rescue MacOSVersionError
          nil
        end

        return if tag_version.blank?

        keys.find do |key|
          key_version = MacOS::Version.from_symbol(key)
          next if key_version.arch != tag_version.arch

          key_version <= tag_version
        rescue MacOSVersionError
          false
        end
      end
    end
  end
end
