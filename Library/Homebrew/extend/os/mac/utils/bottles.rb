module Utils
  class Bottles
    class << self
      undef tag

      def tag
        MacOS.cat
      end
    end

    class Collector
      private

      alias generic_find_matching_tag find_matching_tag

      def find_matching_tag(tag)
        generic_find_matching_tag(tag) ||
          find_older_compatible_tag(tag)
      end

      def tag_without_or_later(tag)
        tag
      end

      # Find a bottle built for a previous version of macOS.
      def find_older_compatible_tag(tag)
        begin
          tag_version = MacOS::Version.from_symbol(tag)
        rescue ArgumentError
          return
        end

        keys.find do |key|
          key_tag_version = tag_without_or_later(key)
          begin
            MacOS::Version.from_symbol(key_tag_version) <= tag_version
          rescue ArgumentError
            false
          end
        end
      end
    end
  end
end
