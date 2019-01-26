class CoreTap < Tap
  # @private
  def initialize
    super "Homebrew", "core"
    @full_name = "Linuxbrew/homebrew-core" unless ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"]
  end
end
