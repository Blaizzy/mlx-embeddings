# frozen_string_literal: true

module HomebrewArgvExtension
  def value(name)
    arg_prefix = "--#{name}="
    flag_with_value = find { |arg| arg.start_with?(arg_prefix) }
    flag_with_value&.delete_prefix(arg_prefix)
  end

  def debug?
    flag?("--debug") || !ENV["HOMEBREW_DEBUG"].nil?
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
