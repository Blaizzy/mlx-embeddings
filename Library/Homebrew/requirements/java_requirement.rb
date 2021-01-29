# typed: false
# frozen_string_literal: true

require "language/java"

# A requirement on Java.
#
# @api private
class JavaRequirement < Requirement
  extend T::Sig

  fatal true

  attr_reader :java_home, :version

  # A strict Java 8 requirement (1.8) should prompt the user to install
  # an OpenJDK 1.8 distribution. Versions newer than Java 8 are not
  # completely backwards compatible, and contain breaking changes such as
  # strong encapsulation of JDK-internal APIs and a modified version scheme
  # (*.0 not 1.*).
  def suggestion
    if fits_latest?
      JAVA_SUGGESTION_MAP.fetch(JAVA_SUGGESTION_MAP.keys.max)
    else
      JAVA_SUGGESTION_MAP.fetch("1.8")
    end
  end

  satisfy build_env: false do
    setup_java
    next false unless @java

    next true
  end

  def initialize(_tags = [])
    odisabled "depends_on :java",
              '"depends_on "openjdk@11", "depends_on "openjdk@8" or "depends_on "openjdk"'
    super
  end

  sig { returns(String) }
  def message
    version_string = " #{@version}" if @version
    s = "Java#{version_string} is required for this software.\n"
    s += suggestion
    s
  end

  sig { returns(String) }
  def inspect
    "#<#{self.class.name}: version=#{@version.inspect} #{tags.inspect}>"
  end

  def display_s
    if @version
      op = if exact_version?
        "="
      else
        ">="
      end
      "#{name.capitalize} #{op} #{version_without_plus}"
    else
      name.capitalize
    end
  end

  private

  CaskSuggestion = Struct.new(:token, :title) do
    extend T::Sig

    sig { returns(String) }
    def to_str
      title_string = " #{title}" if title
      <<~EOS
        Install#{title_string} with Homebrew Cask:
          brew install --cask #{token}
      EOS
    end
  end

  JAVA_SUGGESTION_MAP = {
    "1.8"  => CaskSuggestion.new(
      "homebrew/cask-versions/adoptopenjdk8",
      "AdoptOpenJDK 8",
    ),
    "12.0" => CaskSuggestion.new("adoptopenjdk", "AdoptOpenJDK"),
  }.freeze

  def version_without_plus
    if exact_version?
      @version
    else
      @version[0, @version.length - 1]
    end
  end

  def exact_version?
    @version && @version.to_s[-1] != "+"
  end

  def fits_latest?
    @version.nil? ||
      @version.to_s.end_with?("+") ||
      @version.to_f >= JAVA_SUGGESTION_MAP.keys.max.to_f
  end

  def setup_java
    java = preferred_java
    return unless java

    @java = java
    @java_home = java.parent.parent
  end

  def possible_javas
    javas = []
    javas << Pathname.new(ENV["JAVA_HOME"])/"bin/java" if ENV["JAVA_HOME"]
    javas << which("java")
    javas
  end

  def preferred_java
    possible_javas.find do |java|
      next false unless java&.executable?
      next true unless @version
      next true if satisfies_version(java)
    end
  end

  def env_java_common
    return unless @java_home

    java_home = Pathname.new(@java_home)
    ENV["JAVA_HOME"] = java_home
    ENV.prepend_path "PATH", java_home/"bin"
  end

  def env_oracle_jdk
    return unless @java_home

    java_home = Pathname.new(@java_home)
    return unless (java_home/"include").exist?

    ENV.append_to_cflags "-I#{java_home}/include"
    ENV.append_to_cflags "-I#{java_home}/include/#{oracle_java_os}"
    true
  end

  def oracle_java_os
    nil
  end

  def satisfies_version(java)
    java_version_s = system_command(java, args: ["-version"], print_stderr: false).stderr[/\d+(\.\d+)?/]
    return false unless java_version_s

    java_version = Version.create(java_version_s)
    needed_version = Version.create(version_without_plus)
    if exact_version?
      java_version == needed_version
    else
      java_version >= needed_version
    end
  end
end

require "extend/os/requirements/java_requirement"
