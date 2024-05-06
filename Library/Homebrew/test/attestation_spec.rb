# frozen_string_literal: true

require "diagnostic"

RSpec.describe Homebrew::Attestation do
  let(:fake_gh) { Pathname.new("/extremely/fake/gh") }
  let(:fake_gh_creds) { "fake-gh-api-token" }
  let(:fake_error_status) { instance_double(Process::Status, exitstatus: 1, termsig: nil) }
  let(:fake_auth_status) { instance_double(Process::Status, exitstatus: 4, termsig: nil) }
  let(:cached_download) { "/fake/cached/download" }
  let(:fake_bottle_filename) { instance_double(Bottle::Filename, to_s: "fakebottle--1.0.faketag.bottle.tar.gz") }
  let(:fake_bottle_url) { "https://example.com/#{fake_bottle_filename}" }
  let(:fake_bottle) do
    instance_double(Bottle, cached_download:, filename: fake_bottle_filename, url: fake_bottle_url)
  end
  let(:fake_json_resp) do
    JSON.dump([
      { verificationResult: {
        verifiedTimestamps: [{ timestamp: "2024-03-13T00:00:00Z" }],
        statement:          { subject: [{ name: fake_bottle_filename.to_s }] },
      } },
    ])
  end
  let(:fake_json_resp_backfill) do
    JSON.dump([
      { verificationResult: {
        verifiedTimestamps: [{ timestamp: "2024-03-13T00:00:00Z" }],
        statement:          {
          subject: [{ name: "#{Digest::SHA256.hexdigest(fake_bottle_url)}--#{fake_bottle_filename}" }],
        },
      } },
    ])
  end
  let(:fake_json_resp_too_new) do
    JSON.dump([
      { verificationResult: {
        verifiedTimestamps: [{ timestamp: "2024-03-15T00:00:00Z" }],
        statement:          { subject: [{ name: fake_bottle_filename.to_s }] },
      } },
    ])
  end
  let(:fake_json_resp_wrong_sub) do
    JSON.dump([
      { verificationResult: {
        verifiedTimestamps: [{ timestamp: "2024-03-13T00:00:00Z" }],
        statement:          { subject: [{ name: "wrong-subject.tar.gz" }] },
      } },
    ])
  end

  describe "::gh_executable" do
    it "calls ensure_executable" do
      expect(described_class).to receive(:ensure_executable!)
        .with("gh")
        .and_return(fake_gh)

      described_class.gh_executable
    end
  end

  describe "::check_attestation" do
    before do
      allow(described_class).to receive(:gh_executable)
        .and_return(fake_gh)
    end

    it "raises without any gh credentials" do
      expect(GitHub::API).to receive(:credentials)
        .and_return(nil)

      expect do
        described_class.check_attestation fake_bottle,
                                          described_class::HOMEBREW_CORE_REPO
      end.to raise_error(described_class::GhAuthNeeded)
    end

    it "raises when gh subprocess fails" do
      expect(GitHub::API).to receive(:credentials)
        .and_return(fake_gh_creds)

      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::HOMEBREW_CORE_REPO, "--format", "json")
        .and_raise(ErrorDuringExecution.new(["foo"], status: fake_error_status))

      expect do
        described_class.check_attestation fake_bottle,
                                          described_class::HOMEBREW_CORE_REPO
      end.to raise_error(described_class::InvalidAttestationError)
    end

    it "raises auth error when gh subprocess fails with auth exit code" do
      expect(GitHub::API).to receive(:credentials)
        .and_return(fake_gh_creds)

      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::HOMEBREW_CORE_REPO, "--format", "json")
        .and_raise(ErrorDuringExecution.new(["foo"], status: fake_auth_status))

      expect do
        described_class.check_attestation fake_bottle,
                                          described_class::HOMEBREW_CORE_REPO
      end.to raise_error(described_class::GhAuthNeeded)
    end

    it "raises when gh returns invalid JSON" do
      expect(GitHub::API).to receive(:credentials)
        .and_return(fake_gh_creds)

      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::HOMEBREW_CORE_REPO, "--format", "json")
        .and_return("\"invalid json")

      expect do
        described_class.check_attestation fake_bottle,
                                          described_class::HOMEBREW_CORE_REPO
      end.to raise_error(described_class::InvalidAttestationError)
    end

    it "raises when gh returns other subjects" do
      expect(GitHub::API).to receive(:credentials)
        .and_return(fake_gh_creds)

      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::HOMEBREW_CORE_REPO, "--format", "json")
        .and_return(fake_json_resp_wrong_sub)

      expect do
        described_class.check_attestation fake_bottle,
                                          described_class::HOMEBREW_CORE_REPO
      end.to raise_error(described_class::InvalidAttestationError)
    end
  end

  describe "::check_core_attestation" do
    before do
      allow(described_class).to receive(:gh_executable)
        .and_return(fake_gh)

      allow(GitHub::API).to receive(:credentials)
        .and_return(fake_gh_creds)
    end

    it "calls gh with args for homebrew-core" do
      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::HOMEBREW_CORE_REPO, "--format", "json", "--cert-identity",
              described_class::HOMEBREW_CORE_CI_URI)
        .and_return(fake_json_resp)

      described_class.check_core_attestation fake_bottle
    end

    it "calls gh with args for backfill when homebrew-core fails" do
      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::HOMEBREW_CORE_REPO, "--format", "json", "--cert-identity",
              described_class::HOMEBREW_CORE_CI_URI)
        .once
        .and_raise(described_class::InvalidAttestationError)

      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::BACKFILL_REPO, "--format", "json")
        .and_return(fake_json_resp_backfill)

      described_class.check_core_attestation fake_bottle
    end

    it "raises when the backfilled attestation is too new" do
      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::HOMEBREW_CORE_REPO, "--format", "json", "--cert-identity",
              described_class::HOMEBREW_CORE_CI_URI)
        .once
        .and_raise(described_class::InvalidAttestationError)

      expect(Utils).to receive(:safe_popen_read)
        .with({ "GH_TOKEN" => fake_gh_creds }, fake_gh, "attestation", "verify", cached_download, "--repo",
              described_class::BACKFILL_REPO, "--format", "json")
        .and_return(fake_json_resp_too_new)

      expect do
        described_class.check_core_attestation fake_bottle
      end.to raise_error(described_class::InvalidAttestationError)
    end
  end
end
