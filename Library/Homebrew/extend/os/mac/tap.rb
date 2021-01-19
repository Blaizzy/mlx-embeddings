# typed: true
# frozen_string_literal: true

class Tap
  def self.install_default_cask_tap_if_necessary
    return false if default_cask_tap.installed?

    untapped_official_taps = Homebrew::Settings.read(:untapped)&.split(";") || []
    return false if untapped_official_taps.include?(default_cask_tap.name)

    default_cask_tap.install
    true
  end
end
