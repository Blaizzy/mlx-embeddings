# typed: false
# frozen_string_literal: true

require "bintray"

describe Bintray, :needs_network do
  subject(:bintray) { described_class.new(org: "homebrew") }

  before do
    ENV["HOMEBREW_BINTRAY_USER"] = "BrewTestBot"
    ENV["HOMEBREW_BINTRAY_KEY"] = "deadbeef"
  end

  describe "::remote_checksum" do
    it "detects a published file" do
      hash = bintray.remote_checksum(repo: "bottles", remote_file: "hello-2.10.catalina.bottle.tar.gz")
      expect(hash).to eq("449de5ea35d0e9431f367f1bb34392e450f6853cdccdc6bd04e6ad6471904ddb")
    end

    it "fails on a non-existent file" do
      hash = bintray.remote_checksum(repo: "bottles", remote_file: "my-fake-bottle-1.0.snow_hyena.tar.gz")
      expect(hash).to be nil
    end
  end

  describe "::package_exists?" do
    it "detects a package" do
      results = bintray.package_exists?(repo: "bottles", package: "hello")
      expect(results).to be true
    end
  end
end
