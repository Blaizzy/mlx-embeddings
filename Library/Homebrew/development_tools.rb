# typed: true
# frozen_string_literal: true

require "version"

# @private
class DevelopmentTools
  class << self
    extend T::Sig

    sig { params(tool: String).returns(T.nilable(Pathname)) }
    def locate(tool)
      # Don't call tools (cc, make, strip, etc.) directly!
      # Give the name of the binary you look for as a string to this method
      # in order to get the full path back as a Pathname.
      (@locate ||= {}).fetch(tool) do |key|
        @locate[key] = if File.executable?(path = "/usr/bin/#{tool}")
          Pathname.new path
        # Homebrew GCCs most frequently; much faster to check this before xcrun
        elsif (path = HOMEBREW_PREFIX/"bin/#{tool}").executable?
          path
        end
      end
    end

    sig { returns(T::Boolean) }
    def installed?
      locate("clang").present? || locate("gcc").present?
    end

    sig { returns(String) }
    def installation_instructions
      "Install Clang or run `brew install gcc`."
    end

    sig { returns(String) }
    def custom_installation_instructions
      installation_instructions
    end

    sig { returns(Symbol) }
    def default_compiler
      :clang
    end

    sig { returns(Version) }
    def clang_version
      @clang_version ||= if (path = locate("clang")) &&
                            (build_version = `#{path} --version`[/(?:clang|LLVM) version (\d+\.\d(?:\.\d)?)/, 1])
        Version.new build_version
      else
        Version::NULL
      end
    end

    sig { returns(Version) }
    def clang_build_version
      @clang_build_version ||= if (path = locate("clang")) &&
                                  (build_version = `#{path} --version`[
%r{clang(-| version [^ ]+ \(tags/RELEASE_)(\d{2,})}, 2])
        Version.new build_version
      else
        Version::NULL
      end
    end

    sig { returns(Version) }
    def llvm_clang_build_version
      @llvm_clang_build_version ||= begin
        path = Formulary.factory("llvm").opt_prefix/"bin/clang"
        if path.executable? &&
           (build_version = `#{path} --version`[/clang version (\d+\.\d\.\d)/, 1])
          Version.new build_version
        else
          Version::NULL
        end
      end
    end

    sig { params(cc: String).returns(Version) }
    def gcc_version(cc)
      (@gcc_version ||= {}).fetch(cc) do
        path = HOMEBREW_PREFIX/"opt/#{CompilerSelector.preferred_gcc}/bin"/cc
        path = locate(cc) unless path.exist?
        version = if path &&
                     (build_version = `#{path} --version`[/gcc(?:(?:-\d+(?:\.\d)?)? \(.+\))? (\d+\.\d\.\d)/, 1])
          Version.new build_version
        else
          Version::NULL
        end
        @gcc_version[cc] = version
      end
    end

    sig { void }
    def clear_version_cache
      @clang_version = @clang_build_version = nil
      @gcc_version = {}
    end

    sig { returns(T::Boolean) }
    def build_system_too_old?
      false
    end

    sig { returns(T::Boolean) }
    def system_gcc_too_old?
      false
    end

    sig { returns(T::Boolean) }
    def ca_file_handles_most_https_certificates?
      # The system CA file is too old for some modern HTTPS certificates on
      # older OS versions.
      ENV["HOMEBREW_SYSTEM_CA_CERTIFICATES_TOO_OLD"].nil?
    end

    sig { returns(T::Boolean) }
    def curl_handles_most_https_certificates?
      true
    end

    sig { returns(T::Boolean) }
    def subversion_handles_most_https_certificates?
      true
    end

    sig { returns(T::Hash[String, T.nilable(String)]) }
    def build_system_info
      {
        "os"         => HOMEBREW_SYSTEM,
        "os_version" => OS_VERSION,
        "cpu_family" => Hardware::CPU.family.to_s,
      }
    end
    alias generic_build_system_info build_system_info
  end
end

require "extend/os/development_tools"
