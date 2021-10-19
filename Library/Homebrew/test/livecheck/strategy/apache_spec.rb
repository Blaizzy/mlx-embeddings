# typed: false
# frozen_string_literal: true

require "livecheck/strategy"

describe Homebrew::Livecheck::Strategy::Apache do
  subject(:apache) { described_class }

  let(:apache_urls) {
    urls = {
      version_dir:          "https://www.apache.org/dyn/closer.lua?path=abc/1.2.3/def-1.2.3.tar.gz",
      name_and_version_dir: "https://www.apache.org/dyn/closer.lua?path=abc/def-1.2.3/ghi-1.2.3.tar.gz",
      name_dir_bin:         "https://www.apache.org/dyn/closer.lua?path=abc/def/ghi-1.2.3-bin.tar.gz",
    }

    # Add mirrors.cgi test URLs using the same paths
    urls.clone.each do |key, url|
      next unless url.include?("/closer.lua?path=")

      urls["mirrors_#{key}".to_sym] = url.sub(
        "/closer.lua?path=",
        "/mirrors/mirrors.cgi?action=download&filename=",
      )
    end

    urls
  }
  let(:non_apache_url) { "https://brew.sh/test" }

  let(:generated) {
    {
      version_dir:          {
        url:   "https://archive.apache.org/dist/abc/",
        regex: %r{href=["']?v?(\d+(?:\.\d+)+)/}i,
      },
      name_and_version_dir: {
        url:   "https://archive.apache.org/dist/abc/",
        regex: %r{href=["']?def-v?(\d+(?:\.\d+)+)/}i,
      },
      name_dir_bin:         {
        url:   "https://archive.apache.org/dist/abc/def/",
        regex: /href=["']?ghi-v?(\d+(?:\.\d+)+)-bin\.t/i,
      },
    }
  }

  describe "::match?" do
    it "returns true for an Apache URL" do
      apache_urls.each_value { |url| expect(apache.match?(url)).to be true }
    end

    it "returns false for a non-Apache URL" do
      expect(apache.match?(non_apache_url)).to be false
    end
  end

  describe "::generate_input_values" do
    it "returns a hash containing url and regex for an Apache URL" do
      apache_urls.each do |key, url|
        generated_key = key.to_s.start_with?("mirrors_") ? key.to_s.delete_prefix("mirrors_").to_sym : key
        expect(apache.generate_input_values(url)).to eq(generated[generated_key])
      end
    end

    it "returns an empty hash for a non-Apache URL" do
      expect(apache.generate_input_values(non_apache_url)).to eq({})
    end
  end
end
