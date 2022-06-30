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

    # This regex is used to extract the part of an ETag within quotation marks,
    # ignoring any leading weak validator indicator (`W/`). This simplifies
    # ETag comparison in `#curl_check_http_content`.
    ETAG_VALUE_REGEX = %r{^(?:[wW]/)?"((?:[^"]|\\")*)"}.freeze

    # HTTP responses and body content are typically separated by a double
    # `CRLF` (whereas HTTP header lines are separated by a single `CRLF`).
    # In rare cases, this can also be a double newline (`\n\n`).
    HTTP_RESPONSE_BODY_SEPARATOR = "\r\n\r\n"

    # This regex is used to isolate the parts of an HTTP status line, namely
    # the status code and any following descriptive text (e.g., `Not Found`).
    HTTP_STATUS_LINE_REGEX = %r{^HTTP/.* (?<code>\d+)(?: (?<text>[^\r\n]+))?}.freeze

    private_constant :ETAG_VALUE_REGEX, :HTTP_RESPONSE_BODY_SEPARATOR, :HTTP_STATUS_LINE_REGEX

    module_function

    def curl_executable(use_homebrew_curl: false)
      return HOMEBREW_BREWED_CURL_PATH if use_homebrew_curl

      @curl_executable ||= HOMEBREW_SHIMS_PATH/"shared/curl"
    end

    def curl_path
      @curl_path ||= Utils.popen_read(curl_executable, "--homebrew=print-path").chomp.presence
    end

    def clear_path_cache
      @curl_path = nil
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
      args << "--cookie" << "/dev/null"

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

      result = system_command curl_executable(use_homebrew_curl: use_homebrew_curl),
                              args:    curl_args(*args, **options),
                              env:     env,
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

    def curl_download(*args, to: nil, try_partial: false, **options)
      destination = Pathname(to)
      destination.dirname.mkpath

      if try_partial
        range_stdout = curl_output("--location", "--head", *args, **options).stdout
        parsed_output = parse_curl_output(range_stdout)

        headers = if parsed_output[:responses].present?
          parsed_output[:responses].last[:headers]
        else
          {}
        end

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
    # @param response [Hash] A response hash from `#parse_curl_response`.
    # @return [true, false] Whether a response contains headers indicating that
    #   the URL is protected by Cloudflare.
    sig { params(response: T::Hash[Symbol, T.untyped]).returns(T::Boolean) }
    def url_protected_by_cloudflare?(response)
      return false if response[:headers].blank?
      return false unless [403, 503].include?(response[:status_code].to_i)

      set_cookie_header = Array(response[:headers]["set-cookie"])
      has_cloudflare_cookie_header = set_cookie_header.compact.any? do |cookie|
        cookie.match?(/^(__cfduid|__cf_bm)=/i)
      end

      server_header = Array(response[:headers]["server"])
      has_cloudflare_server = server_header.compact.any? do |server|
        server.match?(/^cloudflare/i)
      end

      has_cloudflare_cookie_header && has_cloudflare_server
    end

    # Check if a URL is protected by Incapsula (e.g. corsair.com).
    # @param response [Hash] A response hash from `#parse_curl_response`.
    # @return [true, false] Whether a response contains headers indicating that
    #   the URL is protected by Incapsula.
    sig { params(response: T::Hash[Symbol, T.untyped]).returns(T::Boolean) }
    def url_protected_by_incapsula?(response)
      return false if response[:headers].blank?
      return false if response[:status_code].to_i != 403

      set_cookie_header = Array(response[:headers]["set-cookie"])
      set_cookie_header.compact.any? { |cookie| cookie.match?(/^(visid_incap|incap_ses)_/i) }
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

          next unless http_status_ok?(secure_details[:status_code])

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
        break if http_status_ok?(details[:status_code])
      end

      unless details[:status_code]
        # Hack around https://github.com/Homebrew/brew/issues/3199
        return if MacOS.version == :el_capitan

        return "The #{url_type} #{url} is not reachable"
      end

      unless http_status_ok?(details[:status_code])
        return if details[:responses].any? do |response|
          url_protected_by_cloudflare?(response) || url_protected_by_incapsula?(response)
        end

        return "The #{url_type} #{url} is not reachable (HTTP status code #{details[:status_code]})"
      end

      if url.start_with?("https://") && Homebrew::EnvConfig.no_insecure_redirect? &&
         (details[:final_url].present? && !details[:final_url].start_with?("https://"))
        return "The #{url_type} #{url} redirects back to HTTP"
      end

      return unless secure_details

      return if !http_status_ok?(details[:status_code]) || !http_status_ok?(secure_details[:status_code])

      etag_match = details[:etag] &&
                   details[:etag] == secure_details[:etag]
      content_length_match =
        details[:content_length] &&
        details[:content_length] == secure_details[:content_length]
      file_match = details[:file_hash] == secure_details[:file_hash]

      http_with_https_available =
        url.start_with?("http://") &&
        (secure_details[:final_url].present? && secure_details[:final_url].start_with?("https://"))

      if (etag_match || content_length_match || file_match) && http_with_https_available
        return "The #{url_type} #{url} should use HTTPS rather than HTTP"
      end

      return unless check_content

      no_protocol_file_contents = %r{https?:\\?/\\?/}
      http_content = details[:file]&.scrub&.gsub(no_protocol_file_contents, "/")
      https_content = secure_details[:file]&.scrub&.gsub(no_protocol_file_contents, "/")

      # Check for the same content after removing all protocols
      if (http_content && https_content) && (http_content == https_content) && http_with_https_available
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

      # Convert specs to options. This is mostly key-value options,
      # unless the value is a boolean in which case treat as as flag.
      specs = specs.flat_map do |option, argument|
        next [] if argument == false # No flag.

        args = ["--#{option.to_s.tr("_", "-")}"]
        args << argument unless argument == true # It's a flag.
        args
      end

      max_time = hash_needed ? 600 : 25
      output, _, status = curl_output(
        *specs, "--dump-header", "-", "--output", file.path, "--location", url,
        use_homebrew_curl: use_homebrew_curl,
        connect_timeout:   15,
        max_time:          max_time,
        retry_max_time:    max_time,
        user_agent:        user_agent
      )

      parsed_output = parse_curl_output(output)
      responses = parsed_output[:responses]

      final_url = curl_response_last_location(responses)
      headers = if responses.last.present?
        status_code = responses.last[:status_code]
        responses.last[:headers]
      else
        {}
      end
      etag = headers["etag"][ETAG_VALUE_REGEX, 1] if headers["etag"].present?
      content_length = headers["content-length"]

      if status.success?
        open_args = {}
        # Try to get encoding from Content-Type header
        # TODO: add guessing encoding by <meta http-equiv="Content-Type" ...> tag
        if (content_type = headers["content-type"]) &&
           (match = content_type.match(/;\s*charset\s*=\s*([^\s]+)/)) &&
           (charset = match[1])
          begin
            open_args[:encoding] = Encoding.find(charset)
          rescue ArgumentError
            # Unknown charset in Content-Type header
          end
        end
        file_contents = File.read(file.path, open_args)
        file_hash = Digest::SHA2.hexdigest(file_contents) if hash_needed
      end

      {
        url:            url,
        final_url:      final_url,
        status_code:    status_code,
        headers:        headers,
        etag:           etag,
        content_length: content_length,
        file:           file_contents,
        file_hash:      file_hash,
        responses:      responses,
      }
    ensure
      file.unlink
    end

    def curl_supports_tls13?
      @curl_supports_tls13 ||= Hash.new do |h, key|
        h[key] = quiet_system(curl_executable, "--tlsv1.3", "--head", "https://brew.sh/")
      end
      @curl_supports_tls13[curl_path]
    end

    def http_status_ok?(status)
      (100..299).cover?(status.to_i)
    end

    # Separates the output text from `curl` into an array of HTTP responses and
    # the final response body (i.e. content). Response hashes contain the
    # `:status_code`, `:status_text`, and `:headers`.
    # @param output [String] The output text from `curl` containing HTTP
    #   responses, body content, or both.
    # @param max_iterations [Integer] The maximum number of iterations for the
    #   `while` loop that parses HTTP response text. This should correspond to
    #   the maximum number of requests in the output. If `curl`'s `--max-redirs`
    #   option is used, `max_iterations` should be `max-redirs + 1`, to
    #   account for any final response after the redirections.
    # @return [Hash] A hash containing an array of response hashes and the body
    #   content, if found.
    sig { params(output: String, max_iterations: Integer).returns(T::Hash[Symbol, T.untyped]) }
    def parse_curl_output(output, max_iterations: 25)
      responses = []

      iterations = 0
      output = output.lstrip
      while output.match?(%r{\AHTTP/[\d.]+ \d+}) && output.include?(HTTP_RESPONSE_BODY_SEPARATOR)
        iterations += 1
        raise "Too many redirects (max = #{max_iterations})" if iterations > max_iterations

        response_text, _, output = output.partition(HTTP_RESPONSE_BODY_SEPARATOR)
        output = output.lstrip
        next if response_text.blank?

        response_text.chomp!
        response = parse_curl_response(response_text)
        responses << response if response.present?
      end

      { responses: responses, body: output }
    end

    # Returns the URL from the last location header found in cURL responses,
    # if any.
    # @param responses [Array<Hash>] An array of hashes containing response
    #   status information and headers from `#parse_curl_response`.
    # @param absolutize [true, false] Whether to make the location URL absolute.
    # @param base_url [String, nil] The URL to use as a base for making the
    #   `location` URL absolute.
    # @return [String, nil] The URL from the last-occurring `location` header
    #   in the responses or `nil` (if no `location` headers found).
    sig {
      params(
        responses:  T::Array[T::Hash[Symbol, T.untyped]],
        absolutize: T::Boolean,
        base_url:   T.nilable(String),
      ).returns(T.nilable(String))
    }
    def curl_response_last_location(responses, absolutize: false, base_url: nil)
      responses.reverse_each do |response|
        next if response[:headers].blank?

        location = response[:headers]["location"]
        next if location.blank?

        absolute_url = URI.join(base_url, location).to_s if absolutize && base_url.present?
        return absolute_url || location
      end

      nil
    end

    private

    # Parses HTTP response text from `curl` output into a hash containing the
    # information from the status line (status code and, optionally,
    # descriptive text) and headers.
    # @param response_text [String] The text of a `curl` response, consisting
    #   of a status line followed by header lines.
    # @return [Hash] A hash containing the response status information and
    #   headers (as a hash with header names as keys).
    sig { params(response_text: String).returns(T::Hash[Symbol, T.untyped]) }
    def parse_curl_response(response_text)
      response = {}
      return response unless response_text.match?(HTTP_STATUS_LINE_REGEX)

      # Parse the status line and remove it
      match = response_text.match(HTTP_STATUS_LINE_REGEX)
      response[:status_code] = match["code"] if match["code"].present?
      response[:status_text] = match["text"] if match["text"].present?
      response_text = response_text.sub(%r{^HTTP/.* (\d+).*$\s*}, "")

      # Create a hash from the header lines
      response[:headers] = {}
      response_text.split("\r\n").each do |line|
        header_name, header_value = line.split(/:\s*/, 2)
        next if header_name.blank?

        header_name = header_name.strip.downcase
        header_value&.strip!

        case response[:headers][header_name]
        when nil
          response[:headers][header_name] = header_value
        when String
          response[:headers][header_name] = [response[:headers][header_name], header_value]
        when Array
          response[:headers][header_name].push(header_value)
        end

        response[:headers][header_name]
      end

      response
    end
  end
end

# FIXME: Include `Utils::Curl` explicitly everywhere it is used.
include Utils::Curl # rubocop:disable Style/MixinUsage
