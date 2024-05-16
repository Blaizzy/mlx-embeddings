# frozen_string_literal: true

RSpec.describe Cask::CaskLoader::FromAPILoader, :cask do
  shared_context "with API setup" do |new_token|
    let(:token) { new_token }
    let(:cask_from_source) { Cask::CaskLoader.load(token) }
    let(:cask_json) do
      hash = cask_from_source.to_hash_with_variations
      json = JSON.pretty_generate(hash)
      JSON.parse(json)
    end
    let(:casks_from_api_hash) { { cask_json["token"] => cask_json.except("token") } }
    let(:api_loader) { described_class.new(token, from_json: cask_json) }

    before do
      allow(Homebrew::API::Cask)
        .to receive(:all_casks)
        .and_return(casks_from_api_hash)
    end
  end

  describe ".can_load?" do
    include_context "with API setup", "test-opera"

    context "when not using the API", :no_api do
      it "returns false" do
        expect(described_class.try_new(token)).to be_nil
      end
    end

    context "when using the API" do
      before do
        ENV.delete("HOMEBREW_NO_INSTALL_FROM_API")
      end

      it "returns a loader for valid token" do
        expect(described_class.try_new(token)).not_to be_nil
      end

      it "returns a loader for valid full name" do
        expect(described_class.try_new("homebrew/cask/#{token}")).not_to be_nil
      end

      it "returns nil for full name with invalid tap" do
        expect(described_class.try_new("homebrew/foo/#{token}")).to be_nil
      end
    end
  end

  describe "#load" do
    shared_examples "loads from API" do |cask_token, caskfile_only|
      include_context "with API setup", cask_token
      let(:cask_from_api) { api_loader.load(config: nil) }

      it "loads from JSON API" do
        expect(cask_from_api).to be_a(Cask::Cask)
        expect(cask_from_api.token).to eq(cask_token)
        expect(cask_from_api.loaded_from_api?).to be(true)
        expect(cask_from_api.caskfile_only?).to be(caskfile_only)
      end
    end

    context "with a binary stanza" do
      include_examples "loads from API", "with-binary", false
    end

    context "with cask dependencies" do
      include_examples "loads from API", "with-depends-on-cask-multiple", false
    end

    context "with formula dependencies" do
      include_examples "loads from API", "with-depends-on-formula-multiple", false
    end

    context "with macos dependencies" do
      include_examples "loads from API", "with-depends-on-macos-array", false
    end

    context "with an installer stanza" do
      include_examples "loads from API", "with-installer-script", false
    end

    context "with uninstall stanzas" do
      include_examples "loads from API", "with-uninstall-multi", false
    end

    context "with a zap stanza" do
      include_examples "loads from API", "with-zap", false
    end

    context "with a preflight stanza" do
      include_examples "loads from API", "with-preflight", true
    end

    context "with an uninstall-preflight stanza" do
      include_examples "loads from API", "with-uninstall-preflight", true
    end

    context "with a postflight stanza" do
      include_examples "loads from API", "with-postflight", true
    end

    context "with an uninstall-postflight stanza" do
      include_examples "loads from API", "with-uninstall-postflight", true
    end

    context "with a language stanza" do
      include_examples "loads from API", "with-languages", true
    end
  end
end
