# typed: false
# frozen_string_literal: true

require "api/analytics"
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
    HOMEBREW_CACHE_API = (HOMEBREW_CACHE/"api").freeze
    MAX_RETRIES = 3

    sig { params(endpoint: String).returns(Hash) }
    def fetch(endpoint)
      return cache[endpoint] if cache.present? && cache.key?(endpoint)

      api_url = "#{API_DOMAIN}/#{endpoint}"
      output = Utils::Curl.curl_output("--fail", api_url, max_time: 5)
      raise ArgumentError, "No file found at #{Tty.underline}#{api_url}#{Tty.reset}" unless output.success?

      cache[endpoint] = JSON.parse(output.stdout)
    rescue JSON::ParserError
      raise ArgumentError, "Invalid JSON file: #{Tty.underline}#{api_url}#{Tty.reset}"
    end

    sig { params(endpoint: String, target: Pathname).returns(Hash) }
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

    sig { params(filepath: String, repo: String, git_head: T.nilable(String)).returns(String) }
    def fetch_file_source(filepath, repo:, git_head: nil)
      git_head ||= "master"
      endpoint = "#{git_head}/#{filepath}"
      return cache[endpoint] if cache.present? && cache.key?(endpoint)

      raw_url = "https://raw.githubusercontent.com/#{repo}/#{endpoint}"
      puts "Fetching #{raw_url}..."
      output = Utils::Curl.curl_output("--fail", raw_url, max_time: 5)
      raise ArgumentError, "No file found at #{Tty.underline}#{raw_url}#{Tty.reset}" unless output.success?

      cache[endpoint] = output.stdout
    end
  end
end
