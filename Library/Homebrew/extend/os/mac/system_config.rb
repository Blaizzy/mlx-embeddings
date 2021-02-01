# typed: true
# frozen_string_literal: true

require "system_command"

module SystemConfig
  class << self
    include SystemCommand::Mixin

    undef describe_homebrew_ruby

    def describe_homebrew_ruby
      s = describe_homebrew_ruby_version

      if RUBY_PATH.to_s.match?(%r{^/System/Library/Frameworks/Ruby\.framework/Versions/[12]\.[089]/usr/bin/ruby})
        s
      else
        "#{s} => #{RUBY_PATH}"
      end
    end

    def xcode
      @xcode ||= if MacOS::Xcode.installed?
        xcode = MacOS::Xcode.version.to_s
        xcode += " => #{MacOS::Xcode.prefix}" unless MacOS::Xcode.default_prefix?
        xcode
      end
    end

    def clt
      @clt ||= MacOS::CLT.version if MacOS::CLT.installed?
    end

    def xquartz
      @xquartz ||= "#{MacOS::XQuartz.version} => #{describe_path(MacOS::XQuartz.prefix)}" if MacOS::XQuartz.installed?
    end

    def dump_verbose_config(f = $stdout)
      dump_generic_verbose_config(f)
      f.puts "macOS: #{MacOS.full_version}-#{kernel}"
      f.puts "CLT: #{clt || "N/A"}"
      f.puts "Xcode: #{xcode || "N/A"}"
      f.puts "XQuartz: #{xquartz}" if xquartz
      f.puts "Rosetta 2: #{Hardware::CPU.in_rosetta2?}" if Hardware::CPU.physical_cpu_arm64?
    end
  end
end
