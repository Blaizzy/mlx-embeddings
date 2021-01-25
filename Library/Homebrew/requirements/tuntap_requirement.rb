# typed: false
# frozen_string_literal: true

require "requirement"

# A requirement on TunTap for macOS.
#
# @api private
class TuntapRequirement < Requirement
  extend T::Sig

  def initialize(tags = [])
    odisabled "depends_on :tuntap"
    super(tags)
  end

  fatal true
  cask "tuntap"
  satisfy(build_env: false) { self.class.binary_tuntap_installed? }

  sig { returns(T::Boolean) }
  def self.binary_tuntap_installed?
    %w[
      /Library/Extensions/tun.kext
      /Library/Extensions/tap.kext
      /Library/LaunchDaemons/net.sf.tuntaposx.tun.plist
      /Library/LaunchDaemons/net.sf.tuntaposx.tap.plist
    ].all? { |file| File.exist?(file) }
  end
end
