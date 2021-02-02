# typed: false
# frozen_string_literal: true

describe Cask::DSL::Appcast do
  subject(:appcast) { described_class.new(url, params) }

  let(:url) { "https://brew.sh" }
  let(:uri) { URI(url) }
  let(:params) { {} }

  describe "#to_s" do
    it "returns the parsed URI string" do
      expect(appcast.to_s).to eq("https://brew.sh")
    end
  end

  describe "#to_yaml" do
    let(:yaml) { [uri, params].to_yaml }

    context "with empty parameters" do
      it "returns an YAML serialized array composed of the URI and parameters" do
        expect(appcast.to_yaml).to eq(yaml)
      end
    end
  end
end
