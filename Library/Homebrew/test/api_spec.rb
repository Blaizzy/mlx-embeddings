# typed: false
# frozen_string_literal: true

require "api"

describe Homebrew::API do
  let(:text) { "foo" }
  let(:json) { '{"foo":"bar"}' }
  let(:json_hash) { JSON.parse(json) }

  def mock_curl_output(stdout: "", success: true)
    curl_output = OpenStruct.new(stdout: stdout, success?: success)
    allow(Utils::Curl).to receive(:curl_output).and_return curl_output
  end

  describe "::fetch" do
    it "fetches a text file" do
      mock_curl_output stdout: text
      fetched_text = described_class.fetch("foo.txt", json: false)
      expect(fetched_text).to eq text
    end

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
end
