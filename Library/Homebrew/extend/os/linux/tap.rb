# typed: true
# frozen_string_literal: true

class CoreTap < Tap
  # @private
  def initialize
    super "Homebrew", "core"
    @full_name = "Homebrew/linuxbrew-core" unless Homebrew::EnvConfig.force_homebrew_on_linux?
  end
end
