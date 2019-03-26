module OS
  # Define OS::Mac on Linux for formula API compatibility.
  module Mac
    module_function

    ::MacOS = self # rubocop:disable Naming/ConstantName

    raise "Loaded OS::Linux on generic OS!" if ENV["HOMEBREW_TEST_GENERIC_OS"]

    def version
      Version::NULL
    end

    def full_version
      Version::NULL
    end

    def languages
      @languages ||= [
        *ARGV.value("language")&.split(","),
        *ENV["HOMEBREW_LANGUAGES"]&.split(","),
        *ENV["LANG"]&.slice(/[a-z]+/),
      ].uniq
    end

    def language
      languages.first
    end

    module Xcode
      module_function

      def version
        Version::NULL
      end
    end
  end
end
