# frozen_string_literal: true

class CoreTap < Tap
  # @private
  def initialize
    super "Homebrew", "core"
    @full_name = "Homebrew/linuxbrew-core" unless ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"]
  end
end
