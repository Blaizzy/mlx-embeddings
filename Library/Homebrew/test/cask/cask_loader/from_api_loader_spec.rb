# typed: false
# frozen_string_literal: true

describe Cask::CaskLoader::FromAPILoader, :cask do
  alias_matcher :be_able_to_load, :be_can_load

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

    context "when not using the API" do
      before do
        allow(Homebrew::EnvConfig)
          .to receive(:no_install_from_api?)
          .and_return(true)
      end

      it "returns false" do
        expect(described_class).not_to be_able_to_load(token)
      end
    end

    context "when using the API" do
      before do
        allow(Homebrew::EnvConfig)
          .to receive(:no_install_from_api?)
          .and_return(false)
      end

      it "returns true for valid token" do
        expect(described_class).to be_able_to_load(token)
      end

      it "returns true for valid full name" do
        expect(described_class).to be_able_to_load("homebrew/cask/#{token}")
      end

      it "returns false for full name with invalid tap" do
        expect(described_class).not_to be_able_to_load("homebrew/cask-versions/#{token}")
      end
    end
  end

  describe "#load" do
    shared_examples "loads from fetched source" do |cask_token|
      include_context "with API setup", cask_token
      let(:content_loader) { instance_double(Cask::CaskLoader::FromContentLoader) }

      it "fetches cask source from API" do
        expect(Homebrew::API::Cask).to receive(:fetch_source).once
        expect(Cask::CaskLoader::FromContentLoader)
          .to receive(:new).once
          .and_return(content_loader)
        expect(content_loader).to receive(:load).once

        api_loader.load(config: nil)
      end
    end

    context "with a preflight stanza" do
      include_examples "loads from fetched source", "with-preflight"
    end

    context "with an uninstall-preflight stanza" do
      include_examples "loads from fetched source", "with-uninstall-preflight"
    end

    context "with a postflight stanza" do
      include_examples "loads from fetched source", "with-postflight"
    end

    context "with an uninstall-postflight stanza" do
      include_examples "loads from fetched source", "with-uninstall-postflight"
    end

    context "with a language stanza" do
      include_examples "loads from fetched source", "with-languages"
    end

    shared_examples "loads from API" do |cask_token|
      include_context "with API setup", cask_token
      let(:cask_from_api) { api_loader.load(config: nil) }

      it "loads from JSON API" do
        expect(Homebrew::API::Cask).not_to receive(:fetch_source)
        expect(Cask::CaskLoader::FromContentLoader).not_to receive(:new)

        expect(cask_from_api).to be_a(Cask::Cask)
        expect(cask_from_api.token).to eq(cask_token)
      end
    end

    context "with a binary stanza" do
      include_examples "loads from API", "with-binary"
    end

    context "with cask dependencies" do
      include_examples "loads from API", "with-depends-on-cask-multiple"
    end

    context "with formula dependencies" do
      include_examples "loads from API", "with-depends-on-formula-multiple"
    end

    context "with macos dependencies" do
      include_examples "loads from API", "with-depends-on-macos-array"
    end

    context "with an installer stanza" do
      include_examples "loads from API", "with-installer-script"
    end

    context "with uninstall stanzas" do
      include_examples "loads from API", "with-uninstall-multi"
    end

    context "with a zap stanza" do
      include_examples "loads from API", "with-zap"
    end
  end
end
