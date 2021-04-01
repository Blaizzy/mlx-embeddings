# typed: true
# frozen_string_literal: true

class DevelopmentTools
  class << self
    extend T::Sig

    def locate(tool)
      (@locate ||= {}).fetch(tool) do |key|
        @locate[key] = if (path = HOMEBREW_PREFIX/"bin/#{tool}").executable?
          path
        elsif File.executable?(path = "/usr/bin/#{tool}")
          Pathname.new path
        end
      end
    end

    sig { returns(Symbol) }
    def default_compiler
      :gcc
    end

    def build_system_info
      brewed_glibc_version = begin
        Formula["glibc"].any_installed_version
      rescue FormulaUnavailableError
        nil
      end
      glibc_version = brewed_glibc_version || OS::Linux::Glibc.system_version
      generic_build_system_info.merge "glibc_version" => glibc_version
    end
  end
end
