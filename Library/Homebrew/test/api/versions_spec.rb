# typed: false
# frozen_string_literal: true

require "api"

describe Homebrew::API::Versions do
  let(:versions_formulae_json) {
    <<~EOS
      {
        "foo":{"version":"1.2.3","revision":0},
        "bar":{"version":"1.2","revision":4}
      }
    EOS
  }
  let(:versions_casks_json) { '{"foo":{"version":"1.2.3"}}' }

  def mock_curl_output(stdout: "", success: true)
    curl_output = OpenStruct.new(stdout: stdout, success?: success)
    allow(Utils::Curl).to receive(:curl_output).and_return curl_output
  end

  describe "::latest_formula_version" do
    it "returns the expected `PkgVersion` when the revision is 0" do
      mock_curl_output stdout: versions_formulae_json
      pkg_version = described_class.latest_formula_version("foo")
      expect(pkg_version.to_s).to eq "1.2.3"
    end

    it "returns the expected `PkgVersion` when the revision is not 0" do
      mock_curl_output stdout: versions_formulae_json
      pkg_version = described_class.latest_formula_version("bar")
      expect(pkg_version.to_s).to eq "1.2_4"
    end

    it "returns `nil` when the formula is not in the JSON file" do
      mock_curl_output stdout: versions_formulae_json
      pkg_version = described_class.latest_formula_version("baz")
      expect(pkg_version).to be_nil
    end
  end

  describe "::latest_cask_version" do
    it "returns the expected `Version`" do
      mock_curl_output stdout: versions_casks_json
      version = described_class.latest_cask_version("foo")
      expect(version.to_s).to eq "1.2.3"
    end

    it "returns `nil` when the cask is not in the JSON file" do
      mock_curl_output stdout: versions_casks_json
      version = described_class.latest_cask_version("bar")
      expect(version).to be_nil
    end
  end
end
