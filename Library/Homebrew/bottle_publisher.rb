# frozen_string_literal: true

require "utils"
require "formula_info"

class BottlePublisher
  def initialize(tap, changed_formulae_names, bintray_org, no_publish, warn_on_publish_failure)
    @tap = tap
    @changed_formulae_names = changed_formulae_names
    @no_publish = no_publish
    @bintray_org = bintray_org
    @warn_on_publish_failure = warn_on_publish_failure
  end

  def publish_and_check_bottles
    # Formulae with affected bottles that were published
    bintray_published_formulae = []

    # Publish bottles on Bintray
    unless @no_publish
      published = publish_changed_formula_bottles
      bintray_published_formulae.concat(published)
    end

    # Verify bintray publishing after all patches have been applied
    bintray_published_formulae.uniq!
    verify_bintray_published(bintray_published_formulae)
  end

  def publish_changed_formula_bottles
    raise "Need to load formulae to publish them!" if ENV["HOMEBREW_DISABLE_LOAD_FORMULA"]

    published = []
    bintray_creds = { user: ENV["HOMEBREW_BINTRAY_USER"], key: ENV["HOMEBREW_BINTRAY_KEY"] }
    if bintray_creds[:user] && bintray_creds[:key]
      @changed_formulae_names.each do |name|
        f = Formula[name]
        next if f.bottle_unneeded? || f.bottle_disabled?

        bintray_org = @bintray_org || @tap.user.downcase
        next unless publish_bottle_file_on_bintray(f, bintray_org, bintray_creds)

        published << f.full_name
      end
    else
      opoo "You must set HOMEBREW_BINTRAY_USER and HOMEBREW_BINTRAY_KEY to add or update bottles on Bintray!"
    end
    published
  end

  # Publishes the current bottle files for a given formula to Bintray
  def publish_bottle_file_on_bintray(f, bintray_org, creds)
    repo = Utils::Bottles::Bintray.repository(f.tap)
    package = Utils::Bottles::Bintray.package(f.name)
    info = FormulaInfo.lookup(f.full_name)
    raise "Failed publishing bottle: failed reading formula info for #{f.full_name}" if info.nil?

    unless info.bottle_info_any
      opoo "No bottle defined in formula #{package}"
      return false
    end
    version = info.pkg_version
    ohai "Publishing on Bintray: #{package} #{version}"
    curl "--write-out", '\n', "--silent", "--fail",
         "--user", "#{creds[:user]}:#{creds[:key]}", "--request", "POST",
         "--header", "Content-Type: application/json",
         "--data", '{"publish_wait_for_secs": 0}',
         "https://api.bintray.com/content/#{bintray_org}/#{repo}/#{package}/#{version}/publish"
    true
  rescue => e
    raise unless @warn_on_publish_failure

    onoe e
    false
  end

  # Verifies that formulae have been published on Bintray by downloading a bottle file
  # for each one. Blocks until the published files are available.
  # Raises an error if the verification fails.
  # This does not currently work for `brew pull`, because it may have cached the old
  # version of a formula.
  def verify_bintray_published(formulae_names)
    return if formulae_names.empty?

    raise "Need to load formulae to verify their publication!" if ENV["HOMEBREW_DISABLE_LOAD_FORMULA"]

    ohai "Verifying bottles published on Bintray"
    formulae = formulae_names.map { |n| Formula[n] }
    max_retries = 300 # shared among all bottles
    poll_retry_delay_seconds = 2

    HOMEBREW_CACHE.cd do
      formulae.each do |f|
        retry_count = 0
        wrote_dots = false
        # Choose arbitrary bottle just to get the host/port for Bintray right
        jinfo = FormulaInfo.lookup(f.full_name)
        unless jinfo
          opoo "Cannot publish bottle: Failed reading info for formula #{f.full_name}"
          next
        end
        bottle_info = jinfo.bottle_info_any
        unless bottle_info
          opoo "No bottle defined in formula #{f.full_name}"
          next
        end

        # Poll for publication completion using a quick partial HEAD, to avoid spurious error messages
        # 401 error is normal while file is still in async publishing process
        url = URI(bottle_info["url"])
        puts "Verifying bottle: #{File.basename(url.path)}"
        http = Net::HTTP.new(url.host, url.port)
        http.use_ssl = true
        retry_count = 0
        http.start do
          loop do
            req = Net::HTTP::Head.new url
            req.initialize_http_header "User-Agent" => HOMEBREW_USER_AGENT_RUBY
            res = http.request req
            break if res.is_a?(Net::HTTPSuccess) || res.code == "302"

            unless res.is_a?(Net::HTTPClientError)
              raise "Failed to find published #{f} bottle at #{url} (#{res.code} #{res.message})!"
            end

            raise "Failed to find published #{f} bottle at #{url}!" if retry_count >= max_retries

            print(wrote_dots ? "." : "Waiting on Bintray.")
            wrote_dots = true
            sleep poll_retry_delay_seconds
            retry_count += 1
          end
        end

        # Actual download and verification
        # We do a retry on this, too, because sometimes the external curl will fail even
        # when the prior HEAD has succeeded.
        puts "\n" if wrote_dots
        filename = File.basename(url.path)
        curl_retry_delay_seconds = 4
        max_curl_retries = 1
        retry_count = 0
        # We're in the cache; make sure to force re-download
        loop do
          curl_download url, to: filename
          break
        rescue
          raise "Failed to download #{f} bottle from #{url}!" if retry_count >= max_curl_retries

          puts "curl download failed; retrying in #{curl_retry_delay_seconds} sec"
          sleep curl_retry_delay_seconds
          curl_retry_delay_seconds *= 2
          retry_count += 1
        end
        checksum = Checksum.new(:sha256, bottle_info["sha256"])
        Pathname.new(filename).verify_checksum(checksum)
      end
    end
  end
end
