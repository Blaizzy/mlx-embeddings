# typed: true
# frozen_string_literal: true

class DevelopmentTools
  class << self
    extend T::Sig

    sig { params(tool: String).returns(T.nilable(Pathname)) }
    def locate(tool)
      (@locate ||= {}).fetch(tool) do |key|
        @locate[key] = if build_system_too_old? &&
                          (binutils_path = HOMEBREW_PREFIX/"opt/binutils/bin/#{tool}").executable?
          binutils_path
        elsif build_system_too_old? && (glibc_path = HOMEBREW_PREFIX/"opt/glibc/bin/#{tool}").executable?
          glibc_path
        elsif (homebrew_path = HOMEBREW_PREFIX/"bin/#{tool}").executable?
          homebrew_path
        elsif File.executable?(system_path = "/usr/bin/#{tool}")
          Pathname.new system_path
        end
      end
    end

    sig { returns(Symbol) }
    def default_compiler
      :gcc
    end

    sig { returns(T::Boolean) }
    def build_system_too_old?
      return @build_system_too_old if defined? @build_system_too_old

      @build_system_too_old = (system_gcc_too_old? || OS::Linux::Glibc.below_ci_version?)
    end

    sig { returns(T::Boolean) }
    def system_gcc_too_old?
      gcc = "/usr/bin/gcc"
      return true unless File.exist?(gcc)

      gcc_version(gcc) < OS::LINUX_GCC_CI_VERSION
    end

    sig { returns(T::Hash[String, T.nilable(String)]) }
    def build_system_info
      generic_build_system_info.merge({
        "glibc_version"     => OS::Linux::Glibc.version.to_s.presence,
        "oldest_cpu_family" => Hardware.oldest_cpu.to_s,
      })
    end
  end
end
