# frozen_string_literal: true

require "delegate"
require "etc"

require "system_command"

# A system user.
#
# @api private
class User < DelegateClass(String)
  # Return whether the user has an active GUI session.
  def gui?
    out, _, status = system_command "who"
    return false unless status.success?

    out.lines
       .map(&:split)
       .any? { |user, type,| user == self && type == "console" }
  end

  # Return the current user.
  def self.current
    return @current if defined?(@current)

    pwuid = Etc.getpwuid(Process.euid)
    return if pwuid.nil?

    @current = new(pwuid.name)
  end
end
