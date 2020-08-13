# frozen_string_literal: true

require "utils/repology"

describe Repology do
  describe "formula_data" do
    it "returns nil for invalid Homebrew Formula" do
      expect(described_class.formula_data("invalidName")).to be_nil
    end
  end

  describe "query_api" do
    it "returns a hash of data" do
      response = described_class.query_api

      expect(response).not_to be_nil
      expect(response).to be_a(Hash)
      expect(response.size).not_to eq(0)
      # first hash in array val should include "repo" key/val pair
      expect(response[response.keys[0]].first).to include("repo")
    end
  end

  describe "single_package_query" do
    it "returns nil for non-existent package" do
      response = described_class.single_package_query("invalidName")

      expect(response).to be_nil
    end

    it "returns a hash for existing package" do
      response = described_class.single_package_query("openclonk")

      expect(response).not_to be_nil
      expect(response).to be_a(Hash)
    end
  end

  describe "parse_api_response" do
    response = described_class.parse_api_response

    it "returns a hash of data" do
      expect(response).not_to be_nil
      expect(response).to be_a(Hash)
    end
  end
end
