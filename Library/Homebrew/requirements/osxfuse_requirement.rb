# frozen_string_literal: true

require "requirement"

# A requirement on FUSE for macOS.
#
# @api private
class OsxfuseRequirement < Requirement
  cask "osxfuse"
  fatal true
end

require "extend/os/requirements/osxfuse_requirement"
