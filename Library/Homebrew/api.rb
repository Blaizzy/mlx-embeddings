# typed: false
# frozen_string_literal: true

require "api/analytics"
require "api/bottle"
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
  end
end
