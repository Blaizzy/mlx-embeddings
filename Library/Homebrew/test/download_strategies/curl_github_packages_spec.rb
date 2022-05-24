# typed: false
# frozen_string_literal: true

require "download_strategy"

describe CurlGitHubPackagesDownloadStrategy do
  subject(:strategy) { described_class.new(url, name, version, **specs) }

  let(:name) { "foo" }
  let(:url) { "https://#{GitHubPackages::URL_DOMAIN}/v2/homebrew/core/spec_test/manifests/1.2.3" }
  let(:version) { "1.2.3" }
  let(:specs) { {} }
  let(:authorization) { nil }

  describe "#fetch" do
    before do
      stub_const("HOMEBREW_GITHUB_PACKAGES_AUTH", authorization) if authorization.present?
      strategy.temporary_path.dirname.mkpath
      FileUtils.touch strategy.temporary_path
    end

    it "calls curl with anonymous authentication headers" do
      expect(strategy).to receive(:system_command).with(
        /curl/,
        hash_including(args: array_including_cons("--header", "Authorization: Bearer QQ==")),
      )
      .at_least(:once)
      .and_return(instance_double(SystemCommand::Result, success?: true, stdout: "", assert_success!: nil))

      strategy.fetch
    end

    context "with Github Packages authentication defined" do
      let(:authorization) { "Bearer dead-beef-cafe" }

      it "calls curl with the provided header value" do
        expect(strategy).to receive(:system_command).with(
          /curl/,
          hash_including(args: array_including_cons("--header", "Authorization: #{authorization}")),
        )
        .at_least(:once)
        .and_return(instance_double(SystemCommand::Result, success?: true, stdout: "", assert_success!: nil))

        strategy.fetch
      end
    end
  end
end
