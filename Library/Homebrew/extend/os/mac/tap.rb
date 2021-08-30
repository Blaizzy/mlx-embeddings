# typed: true
# frozen_string_literal: true

class Tap
  def self.install_default_cask_tap_if_necessary(force: false)
    return false if default_cask_tap.installed?
    return false if ENV["HOMEBREW_INSTALL_FROM_API"].present?
    return false if !force && Tap.untapped_official_taps.include?(default_cask_tap.name)

    default_cask_tap.install
    true
  end
end
