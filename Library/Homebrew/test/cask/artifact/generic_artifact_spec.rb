# typed: false
# frozen_string_literal: true

describe Cask::Artifact::Artifact, :cask do
  let(:cask) { Cask::CaskLoader.load(cask_path("with-generic-artifact")) }

  let(:install_phase) {
    lambda do
      cask.artifacts.select { |a| a.is_a?(described_class) }.each do |artifact|
        artifact.install_phase(command: NeverSudoSystemCommand, force: false)
      end
    end
  }

  let(:source_path) { cask.staged_path.join("Caffeine.app") }
  let(:target_path) { cask.config.appdir.join("Caffeine.app") }

  before do
    InstallHelper.install_without_artifacts(cask)
  end

  context "without target" do
    it "fails to load" do
      expect {
        Cask::CaskLoader.load(cask_path("invalid/invalid-generic-artifact-no-target"))
      }.to raise_error(Cask::CaskInvalidError, /Generic Artifact.*requires.*target/)
    end
  end

  context "with relative target" do
    it "does not fail to load" do
      expect {
        Cask::CaskLoader.load(cask_path("generic-artifact-relative-target"))
      }.not_to raise_error
    end
  end

  context "with user-relative target" do
    it "does not fail to load" do
      expect {
        Cask::CaskLoader.load(cask_path("generic-artifact-user-relative-target"))
      }.not_to raise_error
    end
  end

  it "moves the artifact to the proper directory" do
    install_phase.call

    expect(target_path).to be_a_directory
    expect(source_path).to be_a_symlink
  end

  it "avoids clobbering an existing artifact" do
    target_path.mkpath

    expect {
      install_phase.call
    }.to raise_error(Cask::CaskError)

    expect(source_path).to be_a_directory
    expect(target_path).to be_a_directory
    expect(File.identical?(source_path, target_path)).to be false
  end
end
