# typed: false
# frozen_string_literal: true

require "utils/repology"

describe Repology do
  describe "single_package_query", :needs_network do
    it "returns nil for non-existent package" do
      response = described_class.single_package_query("invalidName", repository: "homebrew")

      expect(response).to be_nil
    end

    it "returns a hash for existing package" do
      response = described_class.single_package_query("openclonk", repository: "homebrew")

      expect(response).to be_nil
      # TODO: uncomment (and remove line above) when we have a fix for Repology
      # `curl` issues
      # expect(response).not_to be_nil
      # expect(response).to be_a(Hash)
    end
  end

  describe "parse_api_response", :needs_network do
    it "returns a hash of data" do
      limit = 1
      response = described_class.parse_api_response(limit, repository: "homebrew")

      expect(response).not_to be_nil
      expect(response).to be_a(Hash)
    end
  end
end
