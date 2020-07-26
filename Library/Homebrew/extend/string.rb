# frozen_string_literal: true

require "active_support/core_ext/object/blank"

# Used by the inreplace function (in `utils.rb`).
class StringInreplaceExtension
  attr_accessor :errors, :inreplace_string

  def initialize(str)
    @inreplace_string = str
    @errors = []
  end

  def self.extended(str)
    str.errors = []
  end

  def sub!(before, after)
    result = inreplace_string.sub!(before, after)
    errors << "expected replacement of #{before.inspect} with #{after.inspect}" unless result
    result
  end

  # Warn if nothing was replaced
  def gsub!(before, after, audit_result = true)
    result = inreplace_string.gsub!(before, after)
    errors << "expected replacement of #{before.inspect} with #{after.inspect}" if audit_result && result.nil?
    result
  end

  # Looks for Makefile style variable definitions and replaces the
  # value with "new_value", or removes the definition entirely.
  def change_make_var!(flag, new_value)
    return if gsub!(/^#{Regexp.escape(flag)}[ \t]*[\\?+:!]?=[ \t]*((?:.*\\\n)*.*)$/, "#{flag}=#{new_value}", false)

    errors << "expected to change #{flag.inspect} to #{new_value.inspect}"
  end

  # Removes variable assignments completely.
  def remove_make_var!(flags)
    Array(flags).each do |flag|
      # Also remove trailing \n, if present.
      unless gsub!(/^#{Regexp.escape(flag)}[ \t]*[\\?+:!]?=(?:.*\\\n)*.*$\n?/, "", false)
        errors << "expected to remove #{flag.inspect}"
      end
    end
  end

  # Finds the specified variable
  def get_make_var(flag)
    inreplace_string[/^#{Regexp.escape(flag)}[ \t]*[\\?+:!]?=[ \t]*((?:.*\\\n)*.*)$/, 1]
  end
end
