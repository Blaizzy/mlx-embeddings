# frozen_string_literal: true

require "delegate"
require "etc"

require "system_command"

class User < DelegateClass(String)
  def gui?
    out, _, status = system_command "who"
    return false unless status.success?

    out.lines
       .map(&:split)
       .any? { |user, type,| user == self && type == "console" }
  end

  def self.current
    return @current if defined?(@current)

    pwuid = Etc.getpwuid(Process.euid)
    return if pwuid.nil?

    @current = new(pwuid.name)
  end
end
