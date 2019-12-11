# frozen_string_literal: true

class SystemConfig
  class << self
    undef describe_java, describe_homebrew_ruby

    def describe_java
      # java_home doesn't exist on all macOSs; it might be missing on older versions.
      return "N/A" unless File.executable? "/usr/libexec/java_home"

      out, _, status = system_command("/usr/libexec/java_home", args: ["--xml", "--failfast"], print_stderr: false)
      return "N/A" unless status.success?

      javas = []
      xml = REXML::Document.new(out)
      REXML::XPath.each(xml, "//key[text()='JVMVersion']/following-sibling::string") do |item|
        javas << item.text
      end
      javas.uniq.join(", ")
    end

    def describe_homebrew_ruby
      s = describe_homebrew_ruby_version

      if !RUBY_PATH.to_s.match?(%r{^/System/Library/Frameworks/Ruby\.framework/Versions/[12]\.[089]/usr/bin/ruby})
        "#{s} => #{RUBY_PATH}"
      else
        s
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
    end
  end
end
