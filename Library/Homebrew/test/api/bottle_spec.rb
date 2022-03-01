# typed: false
# frozen_string_literal: true

require "api"

describe Homebrew::API::Bottle do
  let(:bottle_json) {
    <<~EOS
      {
        "name": "hello",
        "pkg_version": "2.10",
        "rebuild": 0,
        "bottles": {
          "arm64_big_sur": {
            "url": "https://ghcr.io/v2/homebrew/core/hello/blobs/sha256:b3b083db0807ff92c6e289a298f378198354b7727fb9ba9f4d550b8e08f90a60"
          },
          "big_sur": {
            "url": "https://ghcr.io/v2/homebrew/core/hello/blobs/sha256:69489ae397e4645127aa7773211310f81ebb6c99e1f8e3e22c5cdb55333f5408"
          },
          "x86_64_linux": {
            "url": "https://ghcr.io/v2/homebrew/core/hello/blobs/sha256:e6980196298e0a9cfe4fa4e328a71a1869a4d5e1d31c38442150ed784cfc0e29"
          }
        },
        "dependencies": []
      }
    EOS
  }
  let(:bottle_hash) { JSON.parse(bottle_json) }

  def mock_curl_output(stdout: "", success: true)
    curl_output = OpenStruct.new(stdout: stdout, success?: success)
    allow(Utils::Curl).to receive(:curl_output).and_return curl_output
  end

  describe "::fetch" do
    it "fetches the bottle JSON for a formula that exists" do
      mock_curl_output stdout: bottle_json
      fetched_hash = described_class.fetch("foo")
      expect(fetched_hash).to eq bottle_hash
    end

    it "raises an error if the formula does not exist" do
      mock_curl_output success: false
      expect { described_class.fetch("bar") }.to raise_error(ArgumentError, /No file found/)
    end

    it "raises an error if the bottle JSON is invalid" do
      mock_curl_output stdout: "foo"
      expect { described_class.fetch("baz") }.to raise_error(ArgumentError, /Invalid JSON file/)
    end
  end

  describe "::available?" do
    it "returns `true` if `fetch` succeeds" do
      allow(described_class).to receive(:fetch)
      expect(described_class.available?("foo")).to be true
    end

    it "returns `false` if `fetch` fails" do
      allow(described_class).to receive(:fetch).and_raise ArgumentError
      expect(described_class.available?("foo")).to be false
    end
  end

  describe "::fetch_bottles" do
    before do
      ENV["HOMEBREW_INSTALL_FROM_API"] = "1"
      allow(described_class).to receive(:fetch).and_return bottle_hash
    end

    it "fetches bottles if a bottle is available" do
      allow(Utils::Bottles).to receive(:tag).and_return :arm64_big_sur
      expect { described_class.fetch_bottles("hello") }.not_to raise_error
    end

    it "raises an error if no bottle is available" do
      allow(Utils::Bottles).to receive(:tag).and_return :catalina
      expect { described_class.fetch_bottles("hello") }.to raise_error(SystemExit)
    end
  end

  describe "::checksum_from_url" do
    let(:sha256) { "b3b083db0807ff92c6e289a298f378198354b7727fb9ba9f4d550b8e08f90a60" }
    let(:url) { "https://ghcr.io/v2/homebrew/core/hello/blobs/sha256:#{sha256}" }
    let(:non_ghp_url) { "https://formulae.brew.sh/api/formula/hello.json" }

    it "returns the `sha256` for a GitHub packages URL" do
      expect(described_class.checksum_from_url(url)).to eq sha256
    end

    it "returns `nil` for a non-GitHub packages URL" do
      expect(described_class.checksum_from_url(non_ghp_url)).to be_nil
    end
  end
end
