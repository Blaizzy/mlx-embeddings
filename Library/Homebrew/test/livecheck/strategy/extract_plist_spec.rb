# typed: false
# frozen_string_literal: true

require "livecheck/strategy"
require "bundle_version"

describe Homebrew::Livecheck::Strategy::ExtractPlist do
  subject(:extract_plist) { described_class }

  let(:http_url) { "https://brew.sh/blog/" }
  let(:non_http_url) { "ftp://brew.sh/" }

  let(:items) do
    {
      "first"  => extract_plist::Item.new(
        bundle_version: Homebrew::BundleVersion.new(nil, "1.2"),
      ),
      "second" => extract_plist::Item.new(
        bundle_version: Homebrew::BundleVersion.new(nil, "1.2.3"),
      ),
    }
  end

  let(:versions) { ["1.2", "1.2.3"] }

  describe "::match?" do
    it "returns true for an HTTP URL" do
      expect(extract_plist.match?(http_url)).to be true
    end

    it "returns false for a non-HTTP URL" do
      expect(extract_plist.match?(non_http_url)).to be false
    end
  end

  describe "::versions_from_items" do
    it "returns an empty array if Items hash is empty" do
      expect(extract_plist.versions_from_items({})).to eq([])
    end

    it "returns an array of version strings when given Items" do
      expect(extract_plist.versions_from_items(items)).to eq(versions)
    end

    it "returns an array of version strings when given Items and a block" do
      # Returning a string from block
      expect(
        extract_plist.versions_from_items(items) do |items|
          items["first"].version
        end,
      ).to eq(["1.2"])

      # Returning an array of strings from block
      expect(
        extract_plist.versions_from_items(items) do |items|
          items.map do |_key, item|
            item.bundle_version.nice_version
          end
        end,
      ).to eq(versions)
    end

    it "allows a nil return from a block" do
      expect(extract_plist.versions_from_items(items) { next }).to eq([])
    end

    it "errors on an invalid return type from a block" do
      expect { extract_plist.versions_from_items(items) { 123 } }
        .to raise_error(TypeError, Homebrew::Livecheck::Strategy::INVALID_BLOCK_RETURN_VALUE_MSG)
    end
  end
end
