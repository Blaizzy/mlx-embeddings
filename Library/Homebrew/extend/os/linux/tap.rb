# typed: true
# frozen_string_literal: true

class CoreTap < Tap
  # @private
  def initialize
    super "Homebrew", "core"
    @full_name = "Homebrew/linuxbrew-core" if HOMEBREW_CORE_DEFAULT_GIT_REMOTE.include?("Homebrew/linuxbrew-core")
  end
end
