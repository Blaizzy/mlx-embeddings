require "cask/all"

require "test/support/helper/cask/fake_system_command"
require "test/support/helper/cask/install_helper"
require "test/support/helper/cask/never_sudo_system_command"

module Cask
  class Config
    remove_const :DEFAULT_DIRS

    DEFAULT_DIRS = {
      appdir:               Pathname.new(TEST_TMPDIR).join("cask-appdir"),
      prefpanedir:          Pathname.new(TEST_TMPDIR).join("cask-prefpanedir"),
      qlplugindir:          Pathname.new(TEST_TMPDIR).join("cask-qlplugindir"),
      dictionarydir:        Pathname.new(TEST_TMPDIR).join("cask-dictionarydir"),
      fontdir:              Pathname.new(TEST_TMPDIR).join("cask-fontdir"),
      colorpickerdir:       Pathname.new(TEST_TMPDIR).join("cask-colorpickerdir"),
      servicedir:           Pathname.new(TEST_TMPDIR).join("cask-servicedir"),
      input_methoddir:      Pathname.new(TEST_TMPDIR).join("cask-input_methoddir"),
      internet_plugindir:   Pathname.new(TEST_TMPDIR).join("cask-internet_plugindir"),
      audio_unit_plugindir: Pathname.new(TEST_TMPDIR).join("cask-audio_unit_plugindir"),
      vst_plugindir:        Pathname.new(TEST_TMPDIR).join("cask-vst_plugindir"),
      vst3_plugindir:       Pathname.new(TEST_TMPDIR).join("cask-vst3_plugindir"),
      screen_saverdir:      Pathname.new(TEST_TMPDIR).join("cask-screen_saverdir"),
    }.freeze
  end
end

RSpec.shared_context "Homebrew Cask", :needs_macos do
  around do |example|
    third_party_tap = Tap.fetch("third-party", "tap")

    begin
      Cask::Config::DEFAULT_DIRS.values.each(&:mkpath)
      Cask::Config.global.binarydir.mkpath

      Tap.default_cask_tap.tap do |tap|
        FileUtils.mkdir_p tap.path.dirname
        FileUtils.ln_sf TEST_FIXTURE_DIR.join("cask"), tap.path
      end

      third_party_tap.tap do |tap|
        FileUtils.mkdir_p tap.path.dirname
        FileUtils.ln_sf TEST_FIXTURE_DIR.join("third-party"), tap.path
      end

      example.run
    ensure
      FileUtils.rm_rf Cask::Config::DEFAULT_DIRS.values
      FileUtils.rm_rf [Cask::Config.global.binarydir, Cask::Caskroom.path, Cask::Cache.path]
      Tap.default_cask_tap.path.unlink
      third_party_tap.path.unlink
      FileUtils.rm_rf third_party_tap.path.parent
      Cask::Config.clear
    end
  end
end

RSpec.configure do |config|
  config.include_context "Homebrew Cask", :cask
end
