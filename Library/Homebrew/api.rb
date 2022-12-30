# typed: false
# frozen_string_literal: true

require "api/analytics"
require "api/cask"
require "api/cask-source"
require "api/formula"
require "api/versions"
require "extend/cachable"

module Homebrew
  # Helper functions for using Homebrew's formulae.brew.sh API.
  #
  # @api private
  module API
    extend T::Sig

    extend Cachable

    module_function

    API_DOMAIN = "https://formulae.brew.sh/api"
    HOMEBREW_CACHE_API = (HOMEBREW_CACHE/"api").freeze
    MAX_RETRIES = 3

    sig { params(endpoint: String, json: T::Boolean).returns(T.any(String, Hash)) }
    def fetch(endpoint, json: true)
      return cache[endpoint] if cache.present? && cache.key?(endpoint)

      api_url = "#{API_DOMAIN}/#{endpoint}"
      output = Utils::Curl.curl_output("--fail", api_url, max_time: 5)
      raise ArgumentError, "No file found at #{Tty.underline}#{api_url}#{Tty.reset}" unless output.success?

      cache[endpoint] = if json
        JSON.parse(output.stdout)
      else
        output.stdout
      end
    rescue JSON::ParserError
      raise ArgumentError, "Invalid JSON file: #{Tty.underline}#{api_url}#{Tty.reset}"
    end

    def fetch_json_api_file(endpoint, target:)
      retry_count = 0

      url = "#{API_DOMAIN}/#{endpoint}"
      begin
        curl_args = %W[--compressed --silent #{url}]
        curl_args.prepend("--time-cond", target) if target.exist? && !target.empty?
        Utils::Curl.curl_download(*curl_args, to: target, max_time: 5)

        JSON.parse(target.read)
      rescue JSON::ParserError
        target.unlink
        retry_count += 1
        odie "Cannot download non-corrupt #{url}!" if retry_count > MAX_RETRIES

        retry
      end
    end
  end
end
