# frozen_string_literal: true

module OS
  module Mac
    class << self
      module Compat
        def preferred_arch
          odeprecated "MacOS.preferred_arch", "Hardware::CPU.arch (or ideally let the compiler handle it)"
          if Hardware::CPU.is_64_bit?
            Hardware::CPU.arch_64_bit
          else
            Hardware::CPU.arch_32_bit
          end
        end

        def tcc_db
          odeprecated "MacOS.tcc_db"
          @tcc_db ||= Pathname.new("/Library/Application Support/com.apple.TCC/TCC.db")
        end

        def pre_mavericks_accessibility_dotfile
          odeprecated "MacOS.pre_mavericks_accessibility_dotfile"
          @pre_mavericks_accessibility_dotfile ||= Pathname.new("/private/var/db/.AccessibilityAPIEnabled")
        end
      end

      prepend Compat
    end
  end
end
