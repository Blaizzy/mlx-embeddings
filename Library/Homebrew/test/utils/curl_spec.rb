# typed: false
# frozen_string_literal: true

require "utils/curl"

describe "Utils::Curl" do
  describe "curl_args" do
    let(:args) { "foo" }
    let(:user_agent_string) { "Lorem ipsum dolor sit amet" }

    it "returns --disable as the first argument when HOMEBREW_CURLRC is not set" do
      # --disable must be the first argument according to "man curl"
      expect(curl_args(*args).first).to eq("--disable")
    end

    it "doesn't return `--disable` as the first argument when HOMEBREW_CURLRC is set" do
      ENV["HOMEBREW_CURLRC"] = "1"
      expect(curl_args(*args).first).not_to eq("--disable")
    end

    it "uses `--retry 3` when HOMEBREW_CURL_RETRIES is unset" do
      expect(curl_args(*args).join(" ")).to include("--retry 3")
    end

    it "uses the given value for `--retry` when HOMEBREW_CURL_RETRIES is set" do
      ENV["HOMEBREW_CURL_RETRIES"] = "10"
      expect(curl_args(*args).join(" ")).to include("--retry 10")
    end

    it "doesn't use `--retry` when `:retry` == `false`" do
      expect(curl_args(*args, retry: false).join(" ")).not_to include("--retry")
    end

    it "uses `--retry 3` when `:retry` == `true`" do
      expect(curl_args(*args, retry: true).join(" ")).to include("--retry 3")
    end

    it "uses HOMEBREW_USER_AGENT_FAKE_SAFARI when `:user_agent` is `:browser` or `:fake`" do
      expect(curl_args(*args, user_agent: :browser).join(" "))
        .to include("--user-agent #{HOMEBREW_USER_AGENT_FAKE_SAFARI}")
      expect(curl_args(*args, user_agent: :fake).join(" "))
        .to include("--user-agent #{HOMEBREW_USER_AGENT_FAKE_SAFARI}")
    end

    it "uses HOMEBREW_USER_AGENT_CURL when `:user_agent` is `:default` or omitted" do
      expect(curl_args(*args, user_agent: :default).join(" ")).to include("--user-agent #{HOMEBREW_USER_AGENT_CURL}")
      expect(curl_args(*args, user_agent: nil).join(" ")).to include("--user-agent #{HOMEBREW_USER_AGENT_CURL}")
      expect(curl_args(*args).join(" ")).to include("--user-agent #{HOMEBREW_USER_AGENT_CURL}")
    end

    it "uses provided user agent string when `:user_agent` is a `String`" do
      expect(curl_args(*args, user_agent: user_agent_string).join(" "))
        .to include("--user-agent #{user_agent_string}")
    end

    it "uses `--fail` unless `:show_output` is `true`" do
      expect(curl_args(*args, show_output: false).join(" ")).to include("--fail")
      expect(curl_args(*args, show_output: nil).join(" ")).to include("--fail")
      expect(curl_args(*args).join(" ")).to include("--fail")
      expect(curl_args(*args, show_output: true).join(" ")).not_to include("--fail")
    end
  end
end
