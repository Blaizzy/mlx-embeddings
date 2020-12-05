# typed: true
# frozen_string_literal: true

class Tap
  def self.install_default_cask_tap_if_necessary
    return false if default_cask_tap.installed?

    default_cask_tap.install
    true
  end
end
