# typed: false
# frozen_string_literal: true

require "open3"

require "extend/time"

module Utils
  # Helper function for interacting with `curl`.
  #
  # @api private
  module Curl
    extend T::Sig

    using TimeRemaining

    module_function

    def curl_executable(use_homebrew_curl: false)
      return Formula["curl"].opt_bin/"curl" if use_homebrew_curl

      @curl ||= [
        ENV["HOMEBREW_CURL"],
        which("curl"),
        "/usr/bin/curl",
      ].compact.map { |c| Pathname(c) }.find(&:executable?)
      raise "No executable `curl` was found" unless @curl

      @curl
    end

    sig {
      params(
        extra_args:      T.untyped,
        connect_timeout: T.any(Integer, Float, NilClass),
        max_time:        T.any(Integer, Float, NilClass),
        retries:         T.nilable(Integer),
        retry_max_time:  T.any(Integer, Float, NilClass),
        show_output:     T.nilable(T::Boolean),
        user_agent:      T.any(String, Symbol, NilClass),
      ).returns(T::Array[T.untyped])
    }
    def curl_args(
      *extra_args,
      connect_timeout: nil,
      max_time: nil,
      retries: Homebrew::EnvConfig.curl_retries.to_i,
      retry_max_time: nil,
      show_output: false,
      user_agent: nil
    )
      args = []

      # do not load .curlrc unless requested (must be the first argument)
      args << "--disable" unless Homebrew::EnvConfig.curlrc?

      # echo any cookies received on a redirect
      args << "--cookie-jar" << "/dev/null"

      args << "--globoff"

      args << "--show-error"

      args << "--user-agent" << case user_agent
      when :browser, :fake
        HOMEBREW_USER_AGENT_FAKE_SAFARI
      when :default, nil
        HOMEBREW_USER_AGENT_CURL
      when String
        user_agent
      else
        raise TypeError, ":user_agent must be :browser/:fake, :default, or a String"
      end

      args << "--header" << "Accept-Language: en"

      unless show_output == true
        args << "--fail"
        args << "--progress-bar" unless Context.current.verbose?
        args << "--verbose" if Homebrew::EnvConfig.curl_verbose?
        args << "--silent" unless $stdout.tty?
      end

      args << "--connect-timeout" << connect_timeout.round(3) if connect_timeout.present?
      args << "--max-time" << max_time.round(3) if max_time.present?

      # A non-positive integer (e.g., 0) or `nil` will omit this argument
      args << "--retry" << retries if retries&.positive?

      args << "--retry-max-time" << retry_max_time.round if retry_max_time.present?

      args + extra_args
    end

    def curl_with_workarounds(
      *args,
      secrets: nil, print_stdout: nil, print_stderr: nil, debug: nil,
      verbose: nil, env: {}, timeout: nil, use_homebrew_curl: false, **options
    )
      end_time = Time.now + timeout if timeout

      command_options = {
        secrets:      secrets,
        print_stdout: print_stdout,
        print_stderr: print_stderr,
        debug:        debug,
        verbose:      verbose,
      }.compact

      # SSL_CERT_FILE can be incorrectly set by users or portable-ruby and screw
      # with SSL downloads so unset it here.
      result = system_command curl_executable(use_homebrew_curl: use_homebrew_curl),
                              args:    curl_args(*args, **options),
                              env:     { "SSL_CERT_FILE" => nil }.merge(env),
                              timeout: end_time&.remaining,
                              **command_options

      return result if result.success? || !args.exclude?("--http1.1")

      raise Timeout::Error, result.stderr.lines.last.chomp if timeout && result.status.exitstatus == 28

      # Error in the HTTP2 framing layer
      if result.status.exitstatus == 16
        return curl_with_workarounds(
          *args, "--http1.1",
          timeout: end_time&.remaining, **command_options, **options
        )
      end

      # This is a workaround for https://github.com/curl/curl/issues/1618.
      if result.status.exitstatus == 56 # Unexpected EOF
        out = curl_output("-V").stdout

        # If `curl` doesn't support HTTP2, the exception is unrelated to this bug.
        return result unless out.include?("HTTP2")

        # The bug is fixed in `curl` >= 7.60.0.
        curl_version = out[/curl (\d+(\.\d+)+)/, 1]
        return result if Gem::Version.new(curl_version) >= Gem::Version.new("7.60.0")

        return curl_with_workarounds(*args, "--http1.1", **command_options, **options)
      end

      result
    end

    def curl(*args, print_stdout: true, **options)
      result = curl_with_workarounds(*args, print_stdout: print_stdout, **options)
      result.assert_success!
      result
    end

    def parse_headers(headers)
      return {} if headers.blank?

      # Skip status code
      headers.split("\r\n")[1..].to_h do |h|
        name, content = h.split(": ")
        [name.downcase, content]
      end
    end

    def curl_download(*args, to: nil, try_partial: true, **options)
      destination = Pathname(to)
      destination.dirname.mkpath

      if try_partial
        range_stdout = curl_output("--location", "--head", *args, **options).stdout
        headers = parse_headers(range_stdout.split("\r\n\r\n").first)

        # Any value for `accept-ranges` other than none indicates that the server supports partial requests.
        # Its absence indicates no support.
        supports_partial = headers.key?("accept-ranges") && headers["accept-ranges"] != "none"

        if supports_partial &&
           destination.exist? &&
           destination.size == headers["content-length"].to_i
          return # We've already downloaded all the bytes
        end
      end

      args = ["--location", "--remote-time", "--output", destination, *args]
      # continue-at shouldn't be used with servers that don't support partial requests.
      args = ["--continue-at", "-", *args] if destination.exist? && supports_partial

      curl(*args, **options)
    end

    def curl_output(*args, **options)
      curl_with_workarounds(*args, print_stderr: false, show_output: true, **options)
    end

    # Check if a URL is protected by CloudFlare (e.g. badlion.net and jaxx.io).
    def url_protected_by_cloudflare?(details)
      [403, 503].include?(details[:status].to_i) &&
        details[:headers].match?(/^Set-Cookie: __cfduid=/i) &&
        details[:headers].match?(/^Server: cloudflare/i)
    end

    # Check if a URL is protected by Incapsula (e.g. corsair.com).
    def url_protected_by_incapsula?(details)
      details[:status].to_i == 403 &&
        details[:headers].match?(/^Set-Cookie: visid_incap_/i) &&
        details[:headers].match?(/^Set-Cookie: incap_ses_/i)
    end

    def curl_check_http_content(url, url_type, specs: {}, user_agents: [:default],
                                check_content: false, strict: false, use_homebrew_curl: false)
      return unless url.start_with? "http"

      secure_url = url.sub(/\Ahttp:/, "https:")
      secure_details = nil
      hash_needed = false
      if url != secure_url
        user_agents.each do |user_agent|
          secure_details = begin
            curl_http_content_headers_and_checksum(
              secure_url,
              specs:             specs,
              hash_needed:       true,
              use_homebrew_curl: use_homebrew_curl,
              user_agent:        user_agent,
            )
          rescue Timeout::Error
            next
          end

          next unless http_status_ok?(secure_details[:status])

          hash_needed = true
          user_agents = [user_agent]
          break
        end
      end

      details = nil
      user_agents.each do |user_agent|
        details =
          curl_http_content_headers_and_checksum(
            url,
            specs:             specs,
            hash_needed:       hash_needed,
            use_homebrew_curl: use_homebrew_curl,
            user_agent:        user_agent,
          )
        break if http_status_ok?(details[:status])
      end

      unless details[:status]
        # Hack around https://github.com/Homebrew/brew/issues/3199
        return if MacOS.version == :el_capitan

        return "The #{url_type} #{url} is not reachable"
      end

      unless http_status_ok?(details[:status])
        return if url_protected_by_cloudflare?(details) || url_protected_by_incapsula?(details)

        return "The #{url_type} #{url} is not reachable (HTTP status code #{details[:status]})"
      end

      if url.start_with?("https://") && Homebrew::EnvConfig.no_insecure_redirect? &&
         !details[:final_url].start_with?("https://")
        return "The #{url_type} #{url} redirects back to HTTP"
      end

      return unless secure_details

      return if !http_status_ok?(details[:status]) || !http_status_ok?(secure_details[:status])

      etag_match = details[:etag] &&
                   details[:etag] == secure_details[:etag]
      content_length_match =
        details[:content_length] &&
        details[:content_length] == secure_details[:content_length]
      file_match = details[:file_hash] == secure_details[:file_hash]

      if (etag_match || content_length_match || file_match) &&
         secure_details[:final_url].start_with?("https://") &&
         url.start_with?("http://")
        return "The #{url_type} #{url} should use HTTPS rather than HTTP"
      end

      return unless check_content

      no_protocol_file_contents = %r{https?:\\?/\\?/}
      http_content = details[:file]&.gsub(no_protocol_file_contents, "/")
      https_content = secure_details[:file]&.gsub(no_protocol_file_contents, "/")

      # Check for the same content after removing all protocols
      if (http_content && https_content) && (http_content == https_content) &&
         url.start_with?("http://") && secure_details[:final_url].start_with?("https://")
        return "The #{url_type} #{url} should use HTTPS rather than HTTP"
      end

      return unless strict

      # Same size, different content after normalization
      # (typical causes: Generated ID, Timestamp, Unix time)
      if http_content.length == https_content.length
        return "The #{url_type} #{url} may be able to use HTTPS rather than HTTP. Please verify it in a browser."
      end

      lenratio = (100 * https_content.length / http_content.length).to_i
      return unless (90..110).cover?(lenratio)

      "The #{url_type} #{url} may be able to use HTTPS rather than HTTP. Please verify it in a browser."
    end

    def curl_http_content_headers_and_checksum(
      url, specs: {}, hash_needed: false,
      use_homebrew_curl: false, user_agent: :default
    )
      file = Tempfile.new.tap(&:close)

      specs = specs.flat_map { |option, argument| ["--#{option.to_s.tr("_", "-")}", argument] }
      max_time = hash_needed ? 600 : 25
      output, _, status = curl_output(
        *specs, "--dump-header", "-", "--output", file.path, "--location", url,
        use_homebrew_curl: use_homebrew_curl,
        connect_timeout:   15,
        max_time:          max_time,
        retry_max_time:    max_time,
        user_agent:        user_agent
      )

      status_code = :unknown
      while status_code == :unknown || status_code.to_s.start_with?("3")
        headers, _, output = output.partition("\r\n\r\n")
        status_code = headers[%r{HTTP/.* (\d+)}, 1]
        location = headers[/^Location:\s*(.*)$/i, 1]
        final_url = location.chomp if location
      end

      if status.success?
        file_contents = File.read(file.path)
        file_hash = Digest::SHA2.hexdigest(file_contents) if hash_needed
      end

      final_url ||= url

      {
        url:            url,
        final_url:      final_url,
        status:         status_code,
        etag:           headers[%r{ETag: ([wW]/)?"(([^"]|\\")*)"}, 2],
        content_length: headers[/Content-Length: (\d+)/, 1],
        headers:        headers,
        file_hash:      file_hash,
        file:           file_contents,
      }
    ensure
      file.unlink
    end

    def http_status_ok?(status)
      (100..299).cover?(status.to_i)
    end
  end
end

# FIXME: Include `Utils::Curl` explicitly everywhere it is used.
include Utils::Curl # rubocop:disable Style/MixinUsage
