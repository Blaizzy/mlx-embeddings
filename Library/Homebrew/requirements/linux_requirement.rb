# frozen_string_literal: true

# A requirement on Linux.
#
# @api private
class LinuxRequirement < Requirement
  fatal true

  satisfy(build_env: false) { OS.linux? }

  def message
    "Linux is required."
  end
end
