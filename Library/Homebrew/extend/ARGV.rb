# frozen_string_literal: true

module HomebrewArgvExtension
  def named
    # TODO: use @instance variable to ||= cache when moving to CLI::Parser
    self - options_only
  end

  def flags_only
    select { |arg| arg.start_with?("--") }
  end

  def formulae
    require "formula"
    # TODO: use @instance variable to ||= cache when moving to CLI::Parser
    (downcased_unique_named - casks).map do |name|
      if name.include?("/") || File.exist?(name)
        Formulary.factory(name, spec)
      else
        Formulary.find_with_priority(name, spec)
      end
    end.uniq(&:name)
  end

  def casks
    # TODO: use @instance variable to ||= cache when moving to CLI::Parser
    downcased_unique_named.grep HOMEBREW_CASK_TAP_CASK_REGEX
  end

  def value(name)
    arg_prefix = "--#{name}="
    flag_with_value = find { |arg| arg.start_with?(arg_prefix) }
    flag_with_value&.delete_prefix(arg_prefix)
  end

  def force?
    flag? "--force"
  end

  def verbose?
    flag?("--verbose") || !ENV["VERBOSE"].nil? || !ENV["HOMEBREW_VERBOSE"].nil?
  end

  def debug?
    flag?("--debug") || !ENV["HOMEBREW_DEBUG"].nil?
  end

  def quieter?
    flag? "--quieter"
  end

  def interactive?
    flag? "--interactive"
  end

  def keep_tmp?
    include? "--keep-tmp"
  end

  def git?
    flag? "--git"
  end

  def homebrew_developer?
    !ENV["HOMEBREW_DEVELOPER"].nil?
  end

  def skip_or_later_bottles?
    homebrew_developer? && !ENV["HOMEBREW_SKIP_OR_LATER_BOTTLES"].nil?
  end

  def no_sandbox?
    include?("--no-sandbox") || !ENV["HOMEBREW_NO_SANDBOX"].nil?
  end

  def ignore_deps?
    include? "--ignore-dependencies"
  end

  def build_stable?
    !(include?("--HEAD") || include?("--devel"))
  end

  def build_universal?
    include? "--universal"
  end

  def build_bottle?
    include?("--build-bottle")
  end

  def bottle_arch
    arch = value "bottle-arch"
    arch&.to_sym
  end

  def build_from_source?
    switch?("s") || include?("--build-from-source")
  end

  # Whether a given formula should be built from source during the current
  # installation run.
  def build_formula_from_source?(f)
    return false if !build_from_source? && !build_bottle?

    formulae.any? { |argv_f| argv_f.full_name == f.full_name }
  end

  def force_bottle?
    include?("--force-bottle")
  end

  def cc
    value "cc"
  end

  def env
    value "env"
  end

  # If the user passes any flags that trigger building over installing from
  # a bottle, they are collected here and returned as an Array for checking.
  def collect_build_flags
    build_flags = []

    build_flags << "--HEAD" if include?("--HEAD")
    build_flags << "--universal" if build_universal?
    build_flags << "--build-bottle" if build_bottle?
    build_flags << "--build-from-source" if build_from_source?

    build_flags
  end

  private

  def options_only
    select { |arg| arg.start_with?("-") }
  end

  def flag?(flag)
    options_only.include?(flag) || switch?(flag[2, 1])
  end

  # e.g. `foo -ns -i --bar` has three switches: `n`, `s` and `i`
  def switch?(char)
    return false if char.length > 1

    options_only.any? { |arg| arg.scan("-").size == 1 && arg.include?(char) }
  end

  def spec(default = :stable)
    if include?("--HEAD")
      :head
    elsif include?("--devel")
      :devel
    else
      default
    end
  end

  def downcased_unique_named
    # Only lowercase names, not paths, bottle filenames or URLs
    # TODO: use @instance variable to ||= cache when moving to CLI::Parser
    named.map do |arg|
      if arg.include?("/") || arg.end_with?(".tar.gz") || File.exist?(arg)
        arg
      else
        arg.downcase
      end
    end.uniq
  end
end
