# typed: false
# frozen_string_literal: true

require "cask/list"

describe Cask::List, :cask do
  it "lists the installed Casks in a pretty fashion" do
    casks = %w[local-caffeine local-transmission].map { |c| Cask::CaskLoader.load(c) }

    casks.each do |c|
      InstallHelper.install_with_caskfile(c)
    end

    expect {
      described_class.list_casks
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
      described_class.list_casks(one: true)
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
      described_class.list_casks(full_name: true)
    }.to output(<<~EOS).to_stdout
      local-caffeine
      local-transmission
      third-party/tap/third-party-cask
    EOS
  end

  describe "lists versions" do
    let!(:casks) {
      ["local-caffeine",
       "local-transmission"].map(&Cask::CaskLoader.method(:load)).each(&InstallHelper.method(:install_with_caskfile))
    }
    let(:expected_output) {
      <<~EOS
        local-caffeine 1.2.3
        local-transmission 2.61
      EOS
    }

    it "of all installed Casks" do
      expect {
        described_class.list_casks(versions: true)
      }.to output(expected_output).to_stdout
    end

    it "of given Casks" do
      expect {
        described_class.list_casks(*casks, versions: true)
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
        described_class.list_casks(transmission, caffeine)
      }.to output(<<~EOS).to_stdout
        ==> App
        #{transmission.config.appdir.join("Transmission.app")} (#{transmission.config.appdir.join("Transmission.app").abv})
        ==> App
        Missing App: #{caffeine.config.appdir.join("Caffeine.app")}
      EOS
    end
  end
end
