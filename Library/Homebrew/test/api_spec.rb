# typed: false
# frozen_string_literal: true

require "api"

describe Homebrew::API do
  let(:text) { "foo" }
  let(:json) { '{"foo":"bar"}' }
  let(:json_hash) { JSON.parse(json) }
  let(:json_invalid) { '{"foo":"bar"' }

  def mock_curl_output(stdout: "", success: true)
    curl_output = OpenStruct.new(stdout: stdout, success?: success)
    allow(Utils::Curl).to receive(:curl_output).and_return curl_output
  end

  def mock_curl_download(stdout:)
    allow(Utils::Curl).to receive(:curl_download) do |*_args, **kwargs|
      kwargs[:to].write stdout
    end
  end

  describe "::fetch" do
    it "fetches a JSON file" do
      mock_curl_output stdout: json
      fetched_json = described_class.fetch("foo.json")
      expect(fetched_json).to eq json_hash
    end

    it "raises an error if the file does not exist" do
      mock_curl_output success: false
      expect { described_class.fetch("bar.txt") }.to raise_error(ArgumentError, /No file found/)
    end

    it "raises an error if the JSON file is invalid" do
      mock_curl_output stdout: text
      expect { described_class.fetch("baz.txt") }.to raise_error(ArgumentError, /Invalid JSON file/)
    end
  end

  describe "::fetch_json_api_file" do
    let!(:cache_dir) { mktmpdir }

    before do
      (cache_dir/"bar.json").write "tmp"
    end

    it "fetches a JSON file" do
      mock_curl_download stdout: json
      fetched_json = described_class.fetch_json_api_file("foo.json", target: cache_dir/"foo.json")
      expect(fetched_json).to eq json_hash
    end

    it "updates an existing JSON file" do
      mock_curl_download stdout: json
      fetched_json = described_class.fetch_json_api_file("bar.json", target: cache_dir/"bar.json")
      expect(fetched_json).to eq json_hash
    end

    it "raises an error if the JSON file is invalid" do
      mock_curl_download stdout: json_invalid
      expect {
        described_class.fetch_json_api_file("baz.json", target: cache_dir/"baz.json")
      }.to raise_error(SystemExit)
    end
  end

  describe "::fetch_file_source" do
    it "fetches a file" do
      mock_curl_output stdout: json
      fetched_json = described_class.fetch_file_source("foo.json", repo: "Homebrew/homebrew-core", git_head: "master")
      expect(fetched_json).to eq json
    end

    it "raises an error if the file does not exist" do
      mock_curl_output success: false
      expect {
        described_class.fetch_file_source("bar.txt", repo: "Homebrew/homebrew-core", git_head: "master")
      }.to raise_error(ArgumentError, /No file found/)
    end
  end
end
