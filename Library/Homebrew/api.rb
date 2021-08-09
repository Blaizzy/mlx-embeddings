# typed: false
# frozen_string_literal: true

require "api/analytics"
require "api/bottle"
require "api/cask"
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

    sig { params(endpoint: String).returns(T.any(String, Hash)) }
    def fetch(endpoint)
      return cache[endpoint] if cache.present? && cache.key?(endpoint)

      api_url = "#{API_DOMAIN}/#{endpoint}"
      output = Utils::Curl.curl_output("--fail", "--max-time", "5", api_url)
      raise ArgumentError, "No file found at #{Tty.underline}#{api_url}#{Tty.reset}" unless output.success?

      cache[endpoint] = JSON.parse(output.stdout)
    rescue JSON::ParserError
      raise ArgumentError, "Invalid JSON file: #{Tty.underline}#{api_url}#{Tty.reset}"
    end
  end
end
