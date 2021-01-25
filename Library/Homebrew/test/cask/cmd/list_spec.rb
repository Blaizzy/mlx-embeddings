# typed: false
# frozen_string_literal: true

require_relative "shared_examples/invalid_option"

describe Cask::Cmd::List, :cask do
  it_behaves_like "a command that handles invalid options"

  it "lists the installed Casks in a pretty fashion" do
    casks = %w[local-caffeine local-transmission].map { |c| Cask::CaskLoader.load(c) }

    casks.each do |c|
      InstallHelper.install_with_caskfile(c)
    end

    expect {
      described_class.run
    }.to output(<<~EOS).to_stdout
      local-caffeine
      local-transmission
    EOS
  end

  it "lists oneline" do
    casks = %w[
      local-caffeine
      third-party/tap/third-party-cask
      local-transmission
    ].map { |c| Cask::CaskLoader.load(c) }

    casks.each do |c|
      InstallHelper.install_with_caskfile(c)
    end

    expect {
      described_class.run("-1")
    }.to output(<<~EOS).to_stdout
      local-caffeine
      local-transmission
      third-party-cask
    EOS
  end

  it "lists full names" do
    casks = %w[
      local-caffeine
      third-party/tap/third-party-cask
      local-transmission
    ].map { |c| Cask::CaskLoader.load(c) }

    casks.each do |c|
      InstallHelper.install_with_caskfile(c)
    end

    expect {
      described_class.run("--full-name")
    }.to output(<<~EOS).to_stdout
      local-caffeine
      local-transmission
      third-party/tap/third-party-cask
    EOS
  end

  describe "lists versions" do
    let(:casks) { ["local-caffeine", "local-transmission"] }
    let(:expected_output) {
      <<~EOS
        local-caffeine 1.2.3
        local-transmission 2.61
      EOS
    }

    before do
      casks.map(&Cask::CaskLoader.method(:load)).each(&InstallHelper.method(:install_with_caskfile))
    end

    it "of all installed Casks" do
      expect {
        described_class.run("--versions")
      }.to output(expected_output).to_stdout
    end

    it "of given Casks" do
      expect {
        described_class.run("--versions", "local-caffeine", "local-transmission")
      }.to output(expected_output).to_stdout
    end
  end

  describe "lists json" do
    let(:casks) { ["local-caffeine", "local-transmission"] }
    let(:expected_output) {
      <<~EOS
        [{"token":"local-caffeine","name":[],"desc":null,"homepage":"https://brew.sh/","url":"file:///usr/local/Homebrew/Library/Homebrew/test/support/fixtures/cask/caffeine.zip","appcast":null,"version":"1.2.3","installed":"1.2.3","outdated":false,"sha256":"67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94","artifacts":[["Caffeine.app"]],"caveats":null,"depends_on":{},"conflicts_with":null,"container":null,"auto_updates":null},{"token":"local-transmission","name":["Transmission"],"desc":"BitTorrent client","homepage":"https://transmissionbt.com/","url":"file:///usr/local/Homebrew/Library/Homebrew/test/support/fixtures/cask/transmission-2.61.dmg","appcast":null,"version":"2.61","installed":"2.61","outdated":false,"sha256":"e44ffa103fbf83f55c8d0b1bea309a43b2880798dae8620b1ee8da5e1095ec68","artifacts":[["Transmission.app"]],"caveats":null,"depends_on":{},"conflicts_with":null,"container":null,"auto_updates":null}]
      EOS
    }

    before do
      casks.map(&Cask::CaskLoader.method(:load)).each(&InstallHelper.method(:install_with_caskfile))
    end

    it "of all installed Casks" do
      expect {
        described_class.run("--json")
      }.to output(expected_output).to_stdout
    end

    it "of given Casks" do
      expect {
        described_class.run("--json", "local-caffeine", "local-transmission")
      }.to output(expected_output).to_stdout
    end
  end

  describe "given a set of installed Casks" do
    let(:caffeine) { Cask::CaskLoader.load(cask_path("local-caffeine")) }
    let(:transmission) { Cask::CaskLoader.load(cask_path("local-transmission")) }
    let(:casks) { [caffeine, transmission] }

    it "lists the installed files for those Casks" do
      casks.each(&InstallHelper.method(:install_without_artifacts_with_caskfile))

      transmission.artifacts.select { |a| a.is_a?(Cask::Artifact::App) }.each do |artifact|
        artifact.install_phase(command: NeverSudoSystemCommand, force: false)
      end

      expect {
        described_class.run("local-transmission", "local-caffeine")
      }.to output(<<~EOS).to_stdout
        ==> App
        #{transmission.config.appdir.join("Transmission.app")} (#{transmission.config.appdir.join("Transmission.app").abv})
        ==> App
        Missing App: #{caffeine.config.appdir.join("Caffeine.app")}
      EOS
    end
  end
end
