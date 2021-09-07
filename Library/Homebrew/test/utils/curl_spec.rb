# typed: false
# frozen_string_literal: true

require "utils/curl"

describe "Utils::Curl" do
  describe "curl_args" do
    let(:args) { ["foo"] }
    let(:user_agent_string) { "Lorem ipsum dolor sit amet" }

    it "returns `--disable` as the first argument when HOMEBREW_CURLRC is not set" do
      # --disable must be the first argument according to "man curl"
      expect(curl_args(*args).first).to eq("--disable")
    end

    it "doesn't return `--disable` as the first argument when HOMEBREW_CURLRC is set" do
      ENV["HOMEBREW_CURLRC"] = "1"
      expect(curl_args(*args).first).not_to eq("--disable")
    end

    it "uses `--connect-timeout` when `:connect_timeout` is Numeric" do
      expect(curl_args(*args, connect_timeout: 123).join(" ")).to include("--connect-timeout 123")
      expect(curl_args(*args, connect_timeout: 123.4).join(" ")).to include("--connect-timeout 123.4")
      expect(curl_args(*args, connect_timeout: 123.4567).join(" ")).to include("--connect-timeout 123.457")
    end

    it "errors when `:connect_timeout` is not Numeric" do
      expect { curl_args(*args, connect_timeout: "test") }.to raise_error(TypeError)
    end

    it "uses `--max-time` when `:max_time` is Numeric" do
      expect(curl_args(*args, max_time: 123).join(" ")).to include("--max-time 123")
      expect(curl_args(*args, max_time: 123.4).join(" ")).to include("--max-time 123.4")
      expect(curl_args(*args, max_time: 123.4567).join(" ")).to include("--max-time 123.457")
    end

    it "errors when `:max_time` is not Numeric" do
      expect { curl_args(*args, max_time: "test") }.to raise_error(TypeError)
    end

    it "uses `--retry 3` when HOMEBREW_CURL_RETRIES is unset" do
      expect(curl_args(*args).join(" ")).to include("--retry 3")
    end

    it "uses the given value for `--retry` when HOMEBREW_CURL_RETRIES is set" do
      ENV["HOMEBREW_CURL_RETRIES"] = "10"
      expect(curl_args(*args).join(" ")).to include("--retry 10")
    end

    it "uses `--retry` when `:retries` is a positive Integer" do
      expect(curl_args(*args, retries: 5).join(" ")).to include("--retry 5")
    end

    it "doesn't use `--retry` when `:retries` is nil or a non-positive Integer" do
      expect(curl_args(*args, retries: nil).join(" ")).not_to include("--retry")
      expect(curl_args(*args, retries: 0).join(" ")).not_to include("--retry")
      expect(curl_args(*args, retries: -1).join(" ")).not_to include("--retry")
    end

    it "errors when `:retries` is not Numeric" do
      expect { curl_args(*args, retries: "test") }.to raise_error(TypeError)
    end

    it "uses `--retry-max-time` when `:retry_max_time` is Numeric" do
      expect(curl_args(*args, retry_max_time: 123).join(" ")).to include("--retry-max-time 123")
      expect(curl_args(*args, retry_max_time: 123.4).join(" ")).to include("--retry-max-time 123")
    end

    it "errors when `:retry_max_time` is not Numeric" do
      expect { curl_args(*args, retry_max_time: "test") }.to raise_error(TypeError)
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

    it "errors when `:user_agent` is not a String or supported Symbol" do
      expect { curl_args(*args, user_agent: :an_unsupported_symbol) }
        .to raise_error(TypeError, ":user_agent must be :browser/:fake, :default, or a String")
      expect { curl_args(*args, user_agent: 123) }.to raise_error(TypeError)
    end

    it "uses `--fail` unless `:show_output` is `true`" do
      expect(curl_args(*args, show_output: false).join(" ")).to include("--fail")
      expect(curl_args(*args, show_output: nil).join(" ")).to include("--fail")
      expect(curl_args(*args).join(" ")).to include("--fail")
      expect(curl_args(*args, show_output: true).join(" ")).not_to include("--fail")
    end
  end
end
