# typed: false
# frozen_string_literal: true

require "api/analytics"
require "api/cask"
require "api/formula"
require "extend/cachable"

module Homebrew
  # Helper functions for using Homebrew's formulae.brew.sh API.
  #
  # @api private
  module API
    extend T::Sig

    extend Cachable

    module_function

    HOMEBREW_CACHE_API = (HOMEBREW_CACHE/"api").freeze

    sig { params(endpoint: String).returns(Hash) }
    def fetch(endpoint)
      return cache[endpoint] if cache.present? && cache.key?(endpoint)

      api_url = "#{Homebrew::EnvConfig.api_domain}/#{endpoint}"
      output = Utils::Curl.curl_output("--fail", api_url)
      if !output.success? && Homebrew::EnvConfig.api_domain != HOMEBREW_API_DEFAULT_DOMAIN
        # Fall back to the default API domain and try again
        api_url = "#{HOMEBREW_API_DEFAULT_DOMAIN}/#{endpoint}"
        output = Utils::Curl.curl_output("--fail", api_url)
      end
      raise ArgumentError, "No file found at #{Tty.underline}#{api_url}#{Tty.reset}" unless output.success?

      cache[endpoint] = JSON.parse(output.stdout)
    rescue JSON::ParserError
      raise ArgumentError, "Invalid JSON file: #{Tty.underline}#{api_url}#{Tty.reset}"
    end

    sig { params(endpoint: String, target: Pathname).returns(Hash) }
    def fetch_json_api_file(endpoint, target:)
      retry_count = 0
      url = "#{Homebrew::EnvConfig.api_domain}/#{endpoint}"
      default_url = "#{HOMEBREW_API_DEFAULT_DOMAIN}/#{endpoint}"

      # TODO: consider using more of Utils::Curl
      curl_args = %W[
        --compressed
        --speed-limit #{ENV.fetch("HOMEBREW_CURL_SPEED_LIMIT")}
        --speed-time #{ENV.fetch("HOMEBREW_CURL_SPEED_TIME")}
      ]
      curl_args << "--progress-bar" unless Context.current.verbose?
      curl_args << "--verbose" if Homebrew::EnvConfig.curl_verbose?
      curl_args << "--silent" unless $stdout.tty?

      skip_download = target.exist? &&
                      !target.empty? &&
                      (Homebrew::EnvConfig.no_auto_update? ||
                      ((Time.now - Homebrew::EnvConfig.api_auto_update_secs.to_i) < target.mtime))

      begin
        begin
          args = curl_args.dup
          args.prepend("--time-cond", target) if target.exist? && !target.empty?
          unless skip_download
            ohai "Downloading #{url}" if $stdout.tty?
            # Disable retries here, we handle them ourselves below.
            Utils::Curl.curl_download(*args, url, to: target, retries: 0, show_error: false)
          end
        rescue ErrorDuringExecution
          if url == default_url
            raise unless target.exist?
            raise if target.empty?
          elsif retry_count.zero? || !target.exist? || target.empty?
            # Fall back to the default API domain and try again
            # This block will be executed only once, because we set `url` to `default_url`
            url = default_url
            target.unlink if target.exist? && target.empty?
            skip_download = false

            retry
          end

          opoo "#{target.basename}: update failed, falling back to cached version."
        end

        FileUtils.touch target
        JSON.parse(target.read)
      rescue JSON::ParserError
        target.unlink
        retry_count += 1
        skip_download = false
        odie "Cannot download non-corrupt #{url}!" if retry_count > Homebrew::EnvConfig.curl_retries.to_i

        retry
      end
    end

    sig { params(filepath: String, repo: String, git_head: T.nilable(String)).returns(String) }
    def fetch_file_source(filepath, repo:, git_head: nil)
      git_head ||= "master"
      endpoint = "#{git_head}/#{filepath}"
      return cache[endpoint] if cache.present? && cache.key?(endpoint)

      raw_url = "https://raw.githubusercontent.com/#{repo}/#{endpoint}"
      output = Utils::Curl.curl_output("--fail", raw_url)
      raise ArgumentError, "No file found at #{Tty.underline}#{raw_url}#{Tty.reset}" unless output.success?

      cache[endpoint] = output.stdout
    end

    sig { params(json: Hash).returns(Hash) }
    def merge_variations(json)
      if (bottle_tag = ::Utils::Bottles.tag.to_s.presence) &&
         (variations = json["variations"].presence) &&
         (variation = variations[bottle_tag].presence)
        json = json.merge(variation)
      end

      json.except("variations")
    end
  end
end
