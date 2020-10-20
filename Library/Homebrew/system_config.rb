# typed: false
# frozen_string_literal: true

require "hardware"
require "software_spec"
require "development_tools"
require "extend/ENV"
require "system_command"

# Helper module for querying information about the system configuration.
#
# @api private
module SystemConfig
  class << self
    extend T::Sig

    include SystemCommand::Mixin

    def clang
      @clang ||= if DevelopmentTools.installed?
        DevelopmentTools.clang_version
      else
        Version::NULL
      end
    end

    def clang_build
      @clang_build ||= if DevelopmentTools.installed?
        DevelopmentTools.clang_build_version
      else
        Version::NULL
      end
    end

    sig { returns(String) }
    def head
      HOMEBREW_REPOSITORY.git_head || "(none)"
    end

    sig { returns(String) }
    def last_commit
      HOMEBREW_REPOSITORY.git_last_commit || "never"
    end

    sig { returns(String) }
    def origin
      HOMEBREW_REPOSITORY.git_origin || "(none)"
    end

    sig { returns(String) }
    def core_tap_head
      CoreTap.instance.git_head || "(none)"
    end

    sig { returns(String) }
    def core_tap_last_commit
      CoreTap.instance.git_last_commit || "never"
    end

    sig { returns(String) }
    def core_tap_branch
      CoreTap.instance.git_branch || "(none)"
    end

    sig { returns(String) }
    def core_tap_origin
      CoreTap.instance.remote || "(none)"
    end

    sig { returns(String) }
    def describe_clang
      return "N/A" if clang.null?

      clang_build_info = clang_build.null? ? "(parse error)" : clang_build
      "#{clang} build #{clang_build_info}"
    end

    def describe_path(path)
      return "N/A" if path.nil?

      realpath = path.realpath
      if realpath == path
        path
      else
        "#{path} => #{realpath}"
      end
    end

    sig { returns(String) }
    def describe_homebrew_ruby_version
      case RUBY_VERSION
      when /^1\.[89]/, /^2\.0/
        "#{RUBY_VERSION}-p#{RUBY_PATCHLEVEL}"
      else
        RUBY_VERSION
      end
    end

    sig { returns(String) }
    def describe_homebrew_ruby
      "#{describe_homebrew_ruby_version} => #{RUBY_PATH}"
    end

    sig { returns(T.nilable(String)) }
    def hardware
      return if Hardware::CPU.type == :dunno

      "CPU: #{Hardware.cores_as_words}-core #{Hardware::CPU.bits}-bit #{Hardware::CPU.family}"
    end

    sig { returns(String) }
    def kernel
      `uname -m`.chomp
    end

    sig { returns(String) }
    def describe_java
      return "N/A" unless which "java"

      _, err, status = system_command("java", args: ["-version"], print_stderr: false)
      return "N/A" unless status.success?

      err[/java version "([\d._]+)"/, 1] || "N/A"
    end

    sig { returns(String) }
    def describe_git
      return "N/A" unless Utils::Git.available?

      "#{Utils::Git.version} => #{Utils::Git.path}"
    end

    sig { returns(String) }
    def describe_curl
      out, = system_command(curl_executable, args: ["--version"])

      if /^curl (?<curl_version>[\d.]+)/ =~ out
        "#{curl_version} => #{curl_executable}"
      else
        "N/A"
      end
    end

    def core_tap_config(f = $stdout)
      if CoreTap.instance.installed?
        f.puts "Core tap ORIGIN: #{core_tap_origin}"
        f.puts "Core tap HEAD: #{core_tap_head}"
        f.puts "Core tap last commit: #{core_tap_last_commit}"
        f.puts "Core tap branch: #{core_tap_branch}"
      else
        f.puts "Core tap: N/A"
      end
    end

    def homebrew_config(f = $stdout)
      f.puts "HOMEBREW_VERSION: #{HOMEBREW_VERSION}"
      f.puts "ORIGIN: #{origin}"
      f.puts "HEAD: #{head}"
      f.puts "Last commit: #{last_commit}"
    end

    def homebrew_env_config(f = $stdout)
      f.puts "HOMEBREW_PREFIX: #{HOMEBREW_PREFIX}"
      {
        HOMEBREW_REPOSITORY: Homebrew::DEFAULT_REPOSITORY,
        HOMEBREW_CELLAR:     Homebrew::DEFAULT_CELLAR,
      }.freeze.each do |key, default|
        value = Object.const_get(key)
        f.puts "#{key}: #{value}" if value.to_s != default.to_s
      end

      Homebrew::EnvConfig::ENVS.each do |env, hash|
        method_name = Homebrew::EnvConfig.env_method_name(env, hash)

        if hash[:boolean]
          f.puts "#{env}: set" if Homebrew::EnvConfig.send(method_name)
          next
        end

        value = Homebrew::EnvConfig.send(method_name)
        next unless value
        next if (default = hash[:default].presence) && value.to_s == default.to_s

        if ENV.sensitive?(env)
          f.puts "#{env}: set"
        else
          f.puts "#{env}: #{value}"
        end
      end
      f.puts "Homebrew Ruby: #{describe_homebrew_ruby}"
    end

    def host_software_config(f = $stdout)
      f.puts "Clang: #{describe_clang}"
      f.puts "Git: #{describe_git}"
      f.puts "Curl: #{describe_curl}"
      f.puts "Java: #{describe_java}" if describe_java != "N/A"
    end

    def dump_verbose_config(f = $stdout)
      homebrew_config(f)
      core_tap_config(f)
      homebrew_env_config(f)
      f.puts hardware if hardware
      host_software_config(f)
    end
    alias dump_generic_verbose_config dump_verbose_config
  end
end

require "extend/os/system_config"
