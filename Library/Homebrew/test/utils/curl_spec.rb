# frozen_string_literal: true

require "utils/curl"

describe "curl" do
  describe "curl_args" do
    it "returns --disable as the first argument when HOMEBREW_CURLRC is not set" do
      # --disable must be the first argument according to "man curl"
      expect(curl_args("foo").first).to eq("--disable")
    end

    it "doesn't return --disable as the first argument when HOMEBREW_CURLRC is set" do
      ENV["HOMEBREW_CURLRC"] = "1"
      expect(curl_args("foo").first).not_to eq("--disable")
    end

    it "uses `--retry 3` when HOMEBREW_CURL_RETRIES is unset" do
      expect(curl_args("foo").join(" ")).to include("--retry 3")
    end

    it "uses the given value for `--retry` when HOMEBREW_CURL_RETRIES is set" do
      ENV["HOMEBREW_CURL_RETRIES"] = "10"
      expect(curl_args("foo").join(" ")).to include("--retry 10")
    end
  end
end
