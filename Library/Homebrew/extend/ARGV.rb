# frozen_string_literal: true

module HomebrewArgvExtension
  def flags_only
    select { |arg| arg.start_with?("--") }
  end

  def formulae
    require "formula"
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

  def debug?
    flag?("--debug") || !ENV["HOMEBREW_DEBUG"].nil?
  end

  def build_bottle?
    include?("--build-bottle")
  end

  def bottle_arch
    arch = value "bottle-arch"
    arch&.to_sym
  end

  def cc
    value "cc"
  end

  def env
    value "env"
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

  def named
    self - options_only
  end

  def downcased_unique_named
    # Only lowercase names, not paths, bottle filenames or URLs
    named.map do |arg|
      if arg.include?("/") || arg.end_with?(".tar.gz") || File.exist?(arg)
        arg
      else
        arg.downcase
      end
    end.uniq
  end
end
