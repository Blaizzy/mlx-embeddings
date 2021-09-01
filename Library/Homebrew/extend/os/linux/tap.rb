# typed: true
# frozen_string_literal: true

class CoreTap < Tap
  # @private
  def initialize
    super "Homebrew", "core"
    @full_name = "Homebrew/linuxbrew-core" if
      !Homebrew::EnvConfig.force_homebrew_on_linux? &&
      !Homebrew::EnvConfig.force_homebrew_core_repo_on_linux?
  end
end
