# frozen_string_literal: true

require "diagnostic"

RSpec.describe Homebrew::Attestation do
  subject(:attestation) { described_class }

  let(:fake_gh) { Pathname.new("/extremely/fake/gh") }
  let(:fake_json_resp) { JSON.dump({ foo: "bar" }) }
  let(:cached_download) { "/fake/cached/download" }
  let(:fake_bottle) { instance_double(Bottle, cached_download:) }

  describe "::gh_executable" do
    before do
      allow(attestation).to receive(:ensure_executable!)
        .and_return(fake_gh)
    end

    it "returns a path to a gh executable" do
      attestation.gh_executable == fake_gh
    end
  end

  describe "::check_core_attestation" do
    before do
      allow(attestation).to receive(:gh_executable)
        .and_return(fake_gh)

      allow(Utils).to receive(:safe_popen_read)
        .and_return(fake_json_resp)
    end

    it "calls gh with args for homebrew-core" do
      expect(Utils).to receive(:safe_popen_read)
        .with(fake_gh, "attestation", "verify", cached_download, "--repo",
              attestation::HOMEBREW_CORE_REPO, "--format", "json", "--cert-identity",
              attestation::HOMEBREW_CORE_CI_URI)
        .and_return(fake_json_resp)

      attestation.check_core_attestation fake_bottle
    end
  end
end
