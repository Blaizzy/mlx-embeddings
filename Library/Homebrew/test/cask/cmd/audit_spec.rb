# typed: false
# frozen_string_literal: true

require "cask/auditor"

describe Cask::Cmd::Audit, :cask do
  let(:cask) { Cask::Cask.new("cask") }
  let(:result) { { warnings: Set.new, errors: Set.new } }

  describe "selection of Casks to audit" do
    it "audits all Casks if no tokens are given" do
      allow(Cask::Cask).to receive(:to_a).and_return([cask, cask])

      expect(Cask::Auditor).to receive(:audit).twice.and_return(result)

      described_class.run
    end

    it "audits specified Casks if tokens are given" do
      cask_token = "nice-app"
      expect(Cask::CaskLoader).to receive(:load).with(cask_token, any_args).and_return(cask)

      expect(Cask::Auditor).to receive(:audit)
        .with(cask, audit_new_cask: false, quarantine: true, any_named_args: true,
              display_failures_only: false, display_passes: true)
        .and_return(result)

      described_class.run(cask_token)
    end
  end

  it "does not pass anything if no flags are specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_new_cask: false, quarantine: true, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken")
  end

  it "passes `audit_download` if the `--download` flag is specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_download: true, audit_new_cask: false, quarantine: true, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken", "--download")
  end

  it "passes `audit_token_conflicts` if the `--token-conflicts` flag is specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_token_conflicts: true, audit_new_cask: false, quarantine: true, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken", "--token-conflicts")
  end

  it "passes `audit_strict` if the `--strict` flag is specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_strict: true, audit_new_cask: false, quarantine: true, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken", "--strict")
  end

  it "passes `audit_online` if the `--online` flag is specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_online: true, audit_new_cask: false, quarantine: true, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken", "--online")
  end

  it "passes `audit_new_cask` if the `--new-cask` flag is specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_new_cask: true, quarantine: true, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken", "--new-cask")
  end

  it "passes `language` if the `--language` flag is specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_new_cask: false, quarantine: true, language: ["de-AT"], any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken", "--language=de-AT")
  end

  it "passes `quarantine` if the `--no-quarantine` flag is specified" do
    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_new_cask: false, quarantine: false, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken", "--no-quarantine")
  end

  it "passes `quarantine` if the `--no-quarantine` flag is in HOMEBREW_CASK_OPTS" do
    ENV["HOMEBREW_CASK_OPTS"] = "--no-quarantine"

    allow(Cask::CaskLoader).to receive(:load).and_return(cask)
    expect(Cask::Auditor).to receive(:audit)
      .with(cask, audit_new_cask: false, quarantine: false, any_named_args: true,
            display_failures_only: false, display_passes: true)
      .and_return(result)

    described_class.run("casktoken")
  end
end
