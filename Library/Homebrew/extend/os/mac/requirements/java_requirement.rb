# typed: false
# frozen_string_literal: true

class JavaRequirement < Requirement
  env do
    env_java_common
    env_oracle_jdk || env_apple
  end

  private

  undef possible_javas, oracle_java_os

  def possible_javas
    javas = []
    javas << Pathname.new(ENV["JAVA_HOME"])/"bin/java" if ENV["JAVA_HOME"]
    javas << java_home_cmd
    which_java = which("java")
    # /usr/bin/java is a stub on macOS
    javas << which_java if which_java.to_s != "/usr/bin/java"
    javas
  end

  def oracle_java_os
    :darwin
  end

  def java_home_cmd
    odisabled "depends_on :java",
              'depends_on "openjdk@11", depends_on "openjdk@8" or depends_on "openjdk"'
  end

  def env_apple
    ENV.append_to_cflags "-I/System/Library/Frameworks/JavaVM.framework/Versions/Current/Headers/"
  end
end
