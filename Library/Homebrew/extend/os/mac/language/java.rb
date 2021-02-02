# typed: true
# frozen_string_literal: true

module Language
  module Java
    def self.java_home(version = nil)
      find_openjdk_formula(version)&.opt_libexec&.join("openjdk.jdk/Contents/Home")
    end
  end
end
