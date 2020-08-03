# frozen_string_literal: true

require "utils/repology"

describe Repology do
  describe "formula_data", :integration_test do
    it "returns nil for invalid Homebrew Formula" do
      expect(described_class.formula_data("invalidName")).to be_nil
    end

    it "validates Homebrew Formula by name" do
      install_test_formula "testball"
      expect(described_class.formula_data("testball")).not_to be_nil
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

  describe "format_package", :integration_test do
    it "returns nil if package is not a valid formula" do
      invalid_formula_response = described_class.format_package("invalidName", "5.5.5")

      expect(invalid_formula_response).to be_nil
    end

    it "returns hash with data for valid formula" do
      install_test_formula "testball"
      formatted_data = described_class.format_package("testball", "0.1")

      expect(formatted_data).not_to be_nil
      expect(formatted_data).to be_a(Hash)
      expect(formatted_data[:repology_latest_version]).not_to be_nil
      expect(formatted_data[:current_formula_version]).not_to be_nil
      expect(formatted_data[:current_formula_version]).to eq("0.1")
      expect(formatted_data).to include(:livecheck_latest_version)
      expect(formatted_data).to include(:open_pull_requests)
    end
  end
end
