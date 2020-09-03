# frozen_string_literal: true

module Language
  module Java
    class << self
      module Compat
        def java_home_cmd(_version = nil)
          odisabled "Language::Java.java_home_cmd",
                    "Language::Java.java_home or Language::Java.overridable_java_home_env"
        end
      end

      prepend Compat
    end
  end
end
