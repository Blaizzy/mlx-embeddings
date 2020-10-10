# typed: true
# frozen_string_literal: true

module Language
  module Java
    def self.system_java_home_cmd(version = nil)
      version_flag = " --version #{version}" if version
      "/usr/libexec/java_home#{version_flag} --failfast 2>/dev/null"
    end
    private_class_method :system_java_home_cmd

    def self.java_home(version = nil)
      f = find_openjdk_formula(version)
      return f.opt_libexec/"openjdk.jdk/Contents/Home" if f

      cmd = system_java_home_cmd(version)
      path = Utils.popen_read(cmd).chomp

      Pathname.new path if path.present?
    end

    def self.java_home_shell(version = nil)
      f = find_openjdk_formula(version)
      return (f.opt_libexec/"openjdk.jdk/Contents/Home").to_s if f

      "$(#{system_java_home_cmd(version)})"
    end
    private_class_method :java_home_shell
  end
end
