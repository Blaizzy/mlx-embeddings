# frozen_string_literal: true

module Language
  module Java
    class << self
      module Compat
        def java_home_cmd(version = nil)
          odeprecated "Language::Java.java_home_cmd",
                      "Language::Java.java_home or Language::Java.overridable_java_home_env"

          # macOS provides /usr/libexec/java_home, but Linux does not.
          return system_java_home_cmd(version) if OS.mac?

          raise NotImplementedError
        end
      end

      prepend Compat
    end
  end
end
