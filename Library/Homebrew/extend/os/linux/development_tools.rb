# typed: true
# frozen_string_literal: true

class DevelopmentTools
  class << self
    extend T::Sig

    sig { params(tool: String).returns(T.nilable(Pathname)) }
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

    sig { returns(T::Hash[String, T.nilable(String)]) }
    def build_system_info
      generic_build_system_info.merge({
        "glibc_version"     => OS::Linux::Glibc.version.to_s.presence,
        "oldest_cpu_family" => Hardware.oldest_cpu.to_s,
      })
    end
  end
end
