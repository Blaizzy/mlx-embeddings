# frozen_string_literal: true

module Language
  module Java
    def self.system_java_home_cmd(version = nil)
      version_flag = " --version #{version}" if version
      "/usr/libexec/java_home#{version_flag}"
    end
    private_class_method :system_java_home_cmd

    def self.java_home(version = nil)
      cmd = system_java_home_cmd(version)
      Pathname.new Utils.popen_read(cmd).chomp
    end

    # @private
    def self.java_home_shell(version = nil)
      "$(#{system_java_home_cmd(version)})"
    end
  end
end
