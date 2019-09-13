# frozen_string_literal: true

require "delegate"
require "etc"

require "system_command"

class User < DelegateClass(String)
  def self.automation_access?
    return @automation_access if defined?(@automation_access)

    *_, status = system_command "osascript", args: [
      "-e", "with timeout of 0.5 seconds",
      "-e", 'tell application "System Events" to get volume settings',
      "-e", "end timeout"
    ], print_stderr: false

    @automation_access = status.success?
  end

  def self.automation_access_instructions
    "Enable Automation Access for “Terminal > System Events” in " \
    "“System Preferences > Security > Privacy > Automation”."
  end

  def gui?
    out, _, status = system_command "who"
    return false unless status.success?

    out.lines
       .map(&:split)
       .any? { |user, type,| user == self && type == "console" }
  end

  def self.current
    @current ||= new(Etc.getpwuid(Process.euid).name)
  end
end
