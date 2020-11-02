# typed: strict
# frozen_string_literal: true

require "requirement"

# A requirement on FUSE for macOS.
#
# @api private
class OsxfuseRequirement < Requirement
  extend T::Sig
  cask "osxfuse"
  fatal true

  sig { returns(String) }
  def display_s
    "FUSE"
  end
end

require "extend/os/requirements/osxfuse_requirement"
