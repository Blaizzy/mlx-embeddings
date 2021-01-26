# typed: false
# frozen_string_literal: true

require_relative "shared_examples/requires_cask_token"
require_relative "shared_examples/invalid_option"

describe Cask::Cmd::Install, :cask do
  it_behaves_like "a command that requires a Cask token"
  it_behaves_like "a command that handles invalid options"

  it "displays the installation progress" do
    output = Regexp.new <<~EOS
      ==> Downloading file:.*caffeine.zip
      ==> Installing Cask local-caffeine
      ==> Moving App 'Caffeine.app' to '.*Caffeine.app'
      .*local-caffeine was successfully installed!
    EOS

    expect {
      described_class.run("local-caffeine")
    }.to output(output).to_stdout
  end

  it "allows staging and activation of multiple Casks at once" do
    described_class.run("local-transmission", "local-caffeine")
    transmission = Cask::CaskLoader.load(cask_path("local-transmission"))
    caffeine = Cask::CaskLoader.load(cask_path("local-caffeine"))
    expect(transmission).to be_installed
    expect(transmission.config.appdir.join("Transmission.app")).to be_a_directory
    expect(caffeine).to be_installed
    expect(caffeine.config.appdir.join("Caffeine.app")).to be_a_directory
  end

  it "recognizes the --appdir flag" do
    appdir = mktmpdir

    expect(Cask::CaskLoader).to receive(:load).with("local-caffeine", any_args)
      .and_wrap_original { |f, *args|
        caffeine = f.call(*args)
        expect(caffeine.config.appdir).to eq appdir
        caffeine
      }

    described_class.run("local-caffeine", "--appdir=#{appdir}")
  end

  it "recognizes the --appdir flag from HOMEBREW_CASK_OPTS" do
    appdir = mktmpdir

    expect(Cask::CaskLoader).to receive(:load).with("local-caffeine", any_args)
      .and_wrap_original { |f, *args|
        caffeine = f.call(*args)
        expect(caffeine.config.appdir).to eq appdir
        caffeine
      }

    ENV["HOMEBREW_CASK_OPTS"] = "--appdir=#{appdir}"

    described_class.run("local-caffeine")
  end

  it "prefers an explicit --appdir flag to one from HOMEBREW_CASK_OPTS" do
    global_appdir = mktmpdir
    appdir = mktmpdir

    expect(Cask::CaskLoader).to receive(:load).with("local-caffeine", any_args)
      .and_wrap_original { |f, *args|
        caffeine = f.call(*args)
        expect(caffeine.config.appdir).to eq appdir
        caffeine
      }

    ENV["HOMEBREW_CASK_OPTS"] = "--appdir=#{global_appdir}"

    described_class.run("local-caffeine", "--appdir=#{appdir}")
  end

  it "skips double install (without nuking existing installation)" do
    described_class.run("local-transmission")
    described_class.run("local-transmission")
    expect(Cask::CaskLoader.load(cask_path("local-transmission"))).to be_installed
  end

  it "prints a warning message on double install" do
    described_class.run("local-transmission")

    expect {
      described_class.run("local-transmission")
    }.to output(/Warning: Cask 'local-transmission' is already installed./).to_stderr
  end

  it "allows double install with --force" do
    described_class.run("local-transmission")

    expect {
      expect {
        described_class.run("local-transmission", "--force")
      }.to output(/It seems there is already an App at.*overwriting\./).to_stderr
    }.to output(/local-transmission was successfully installed!/).to_stdout
  end

  it "skips dependencies with --skip-cask-deps" do
    described_class.run("with-depends-on-cask-multiple", "--skip-cask-deps")
    expect(Cask::CaskLoader.load(cask_path("with-depends-on-cask-multiple"))).to be_installed
    expect(Cask::CaskLoader.load(cask_path("local-caffeine"))).not_to be_installed
    expect(Cask::CaskLoader.load(cask_path("local-transmission"))).not_to be_installed
  end

  it "properly handles Casks that are not present" do
    expect {
      described_class.run("notacask")
    }.to raise_error(Cask::CaskUnavailableError)
  end

  it "returns a suggestion for a misspelled Cask" do
    expect {
      described_class.run("localcaffeine")
    }.to raise_error(
      Cask::CaskUnavailableError,
      "Cask 'localcaffeine' is unavailable: No Cask with this name exists. "\
      "Did you mean 'local-caffeine'?",
    )
  end

  it "returns multiple suggestions for a Cask fragment" do
    expect {
      described_class.run("local")
    }.to raise_error(
      Cask::CaskUnavailableError,
      "Cask 'local' is unavailable: No Cask with this name exists. " \
      "Did you mean one of these?\nlocal-caffeine\nlocal-transmission\n",
    )
  end
end
