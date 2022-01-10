# typed: true
# frozen_string_literal: true

module OS
  # Helper module for querying system information on Linux.
  module Linux
    extend T::Sig

    module_function

    sig { returns(String) }
    def os_version
      if which("lsb_release")
        lsb_info = Utils.popen_read("lsb_release", "-a")
        description = lsb_info[/^Description:\s*(.*)$/, 1].force_encoding("UTF-8")
        codename = lsb_info[/^Codename:\s*(.*)$/, 1]
        if codename.blank? || (codename == "n/a")
          description
        else
          "#{description} (#{codename})"
        end
      elsif (redhat_release = Pathname.new("/etc/redhat-release")).readable?
        redhat_release.read.chomp
      else
        "Unknown"
      end
    end
  end

  # rubocop:disable Style/Documentation
  module Mac
    module_function

    # rubocop:disable Naming/ConstantName
    # rubocop:disable Style/MutableConstant
    ::MacOS = OS::Mac
    # rubocop:enable Naming/ConstantName
    # rubocop:enable Style/MutableConstant

    raise "Loaded OS::Linux on generic OS!" if ENV["HOMEBREW_TEST_GENERIC_OS"]

    def version
      ::Version::NULL
    end

    def full_version
      ::Version::NULL
    end

    def languages
      @languages ||= Array(ENV["LANG"]&.slice(/[a-z]+/)).uniq
    end

    def language
      languages.first
    end

    def sdk_root_needed?
      false
    end

    def sdk_path_if_needed(_v = nil)
      nil
    end

    def sdk_path
      nil
    end

    module Xcode
      module_function

      def version
        ::Version::NULL
      end

      def installed?
        false
      end
    end

    module CLT
      module_function

      def version
        ::Version::NULL
      end

      def installed?
        false
      end
    end
  end
  # rubocop:enable Style/Documentation
end
