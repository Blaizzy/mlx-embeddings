# frozen_string_literal: true

module OS
  module Mac
    class << self
      module Compat
        def preferred_arch
          odisabled "MacOS.preferred_arch", "Hardware::CPU.arch (or ideally let the compiler handle it)"
        end

        def tcc_db
          odisabled "MacOS.tcc_db"
        end

        def pre_mavericks_accessibility_dotfile
          odisabled "MacOS.pre_mavericks_accessibility_dotfile"
        end
      end

      prepend Compat
    end
  end
end
