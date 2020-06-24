# frozen_string_literal: true

module Utils
  class Bottles
    class << self
      undef tag

      def tag
        tag = MacOS.version.to_sym
        tag = "#{tag}_arm".to_sym if Hardware::CPU.arm?
        tag
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
          return
        end

        keys.find do |key|
          MacOS::Version.from_symbol(key) <= tag_version
        rescue MacOSVersionError
          false
        end
      end
    end
  end
end
