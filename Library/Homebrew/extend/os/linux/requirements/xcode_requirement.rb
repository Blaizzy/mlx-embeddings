# typed: false
# frozen_string_literal: true

require "requirement"

class XcodeRequirement < Requirement
  def xcode_installed_version
    true
  end
end
