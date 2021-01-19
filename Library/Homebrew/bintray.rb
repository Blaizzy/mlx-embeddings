# typed: false
# frozen_string_literal: true

require "utils/curl"
require "json"

# Bintray API client.
#
# @api private
class Bintray
  extend T::Sig

  include Context
  include Utils::Curl

  API_URL = "https://api.bintray.com"

  class Error < RuntimeError
  end

  sig { returns(String) }
  def inspect
    "#<Bintray: org=#{@bintray_org}>"
  end

  sig { params(org: T.nilable(String)).void }
  def initialize(org: "homebrew")
    @bintray_org = org

    raise UsageError, "Must set a Bintray organisation!" unless @bintray_org

    ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"] = "1" if @bintray_org == "homebrew" && !OS.mac?
  end

  def open_api(url, *args, auth: true)
    if auth
      raise UsageError, "HOMEBREW_BINTRAY_USER is unset." unless (user = Homebrew::EnvConfig.bintray_user)
      raise UsageError, "HOMEBREW_BINTRAY_KEY is unset." unless (key = Homebrew::EnvConfig.bintray_key)

      args += ["--user", "#{user}:#{key}"]
    end

    curl(*args, url, print_stdout: false, secrets: key)
  end

  sig {
    params(local_file:    String,
           repo:          String,
           package:       String,
           version:       String,
           remote_file:   String,
           sha256:        T.nilable(String),
           warn_on_error: T.nilable(T::Boolean)).void
  }
  def upload(local_file, repo:, package:, version:, remote_file:, sha256: nil, warn_on_error: false)
    unless File.exist? local_file
      msg = "#{local_file} for upload doesn't exist!"
      raise Error, msg unless warn_on_error

      # Warn and return early here since we know this upload is going to fail.
      opoo msg
      return
    end

    url = "#{API_URL}/content/#{@bintray_org}/#{repo}/#{package}/#{version}/#{remote_file}"
    args = ["--upload-file", local_file]
    args += ["--header", "X-Checksum-Sha2: #{sha256}"] if sha256.present?
    args << "--fail" unless warn_on_error

    result = T.unsafe(self).open_api(url, *args)

    json = JSON.parse(result.stdout)
    return if json["message"] == "success"

    msg = "Bottle upload failed: #{json["message"]}"
    raise msg unless warn_on_error

    opoo msg
  end

  sig {
    params(repo:          String,
           package:       String,
           version:       String,
           file_count:    T.nilable(Integer),
           warn_on_error: T.nilable(T::Boolean)).void
  }
  def publish(repo:, package:, version:, file_count:, warn_on_error: false)
    url = "#{API_URL}/content/#{@bintray_org}/#{repo}/#{package}/#{version}/publish"
    upload_args = %w[--request POST]
    upload_args += ["--fail"] unless warn_on_error
    result = T.unsafe(self).open_api(url, *upload_args)
    json = JSON.parse(result.stdout)
    if file_count.present? && json["files"] != file_count
      message = "Bottle publish failed: expected #{file_count} bottles, but published #{json["files"]} instead."
      raise message unless warn_on_error

      opoo message
    end

    odebug "Published #{json["files"]} bottles"
  end

  sig { params(org: T.nilable(String)).returns(T::Boolean) }
  def official_org?(org: @bintray_org)
    %w[homebrew linuxbrew].include? org
  end

  sig { params(url: String).returns(T::Boolean) }
  def stable_mirrored?(url)
    headers, = curl_output("--connect-timeout", "15", "--location", "--head", url)
    status_code = headers.scan(%r{^HTTP/.* (\d+)}).last.first
    status_code.start_with?("2")
  end

  sig {
    params(formula:         Formula,
           repo:            String,
           publish_package: T::Boolean,
           warn_on_error:   T::Boolean).returns(String)
  }
  def mirror_formula(formula, repo: "mirror", publish_package: false, warn_on_error: false)
    package = Utils::Bottles::Bintray.package formula.name

    create_package(repo: repo, package: package) unless package_exists?(repo: repo, package: package)

    formula.downloader.fetch

    version = ERB::Util.url_encode(formula.pkg_version)
    filename = ERB::Util.url_encode(formula.downloader.basename)
    destination_url = "https://dl.bintray.com/#{@bintray_org}/#{repo}/#{filename}"

    odebug "Uploading to #{destination_url}"

    upload(
      formula.downloader.cached_location,
      repo:          repo,
      package:       package,
      version:       version,
      sha256:        formula.stable.checksum,
      remote_file:   filename,
      warn_on_error: warn_on_error,
    )
    return destination_url unless publish_package

    odebug "Publishing #{@bintray_org}/#{repo}/#{package}/#{version}"
    publish(repo: repo, package: package, version: version, file_count: 1, warn_on_error: warn_on_error)

    destination_url
  end

  sig { params(repo: String, package: String).void }
  def create_package(repo:, package:)
    url = "#{API_URL}/packages/#{@bintray_org}/#{repo}"
    data = { name: package, public_download_numbers: true }
    data[:public_stats] = official_org?
    open_api(url, "--header", "Content-Type: application/json", "--request", "POST", "--data", data.to_json)
  end

  sig { params(repo: String, package: String).returns(T::Boolean) }
  def package_exists?(repo:, package:)
    url = "#{API_URL}/packages/#{@bintray_org}/#{repo}/#{package}"
    begin
      open_api(url, "--fail", "--silent", "--output", "/dev/null", auth: false)
    rescue ErrorDuringExecution => e
      stderr = e.output
                .select { |type,| type == :stderr }
                .map { |_, line| line }
                .join
      raise if e.status.exitstatus != 22 && stderr.exclude?("404 Not Found")

      false
    else
      true
    end
  end

  # Gets the SHA-256 checksum of the specified remote file.
  #
  # @return the checksum, the empty string (if the file doesn't have a checksum), nil (if the file doesn't exist)
  sig { params(repo: String, remote_file: String).returns(T.nilable(String)) }
  def remote_checksum(repo:, remote_file:)
    url = "https://dl.bintray.com/#{@bintray_org}/#{repo}/#{remote_file}"
    result = curl_output "--fail", "--silent", "--head", url
    if result.success?
      result.stdout.match(/^X-Checksum-Sha2:\s+(\h{64})\b/i)&.values_at(1)&.first || ""
    else
      raise Error if result.status.exitstatus != 22 && result.stderr.exclude?("404 Not Found")

      nil
    end
  end

  sig { params(bintray_repo: String, bintray_package: String, filename: String).returns(String) }
  def file_delete_instructions(bintray_repo, bintray_package, filename)
    <<~EOS
      Remove this file manually in your web browser:
        https://bintray.com/#{@bintray_org}/#{bintray_repo}/#{bintray_package}/view#files
      Or run:
        curl -X DELETE -u $HOMEBREW_BINTRAY_USER:$HOMEBREW_BINTRAY_KEY \\
        https://api.bintray.com/content/#{@bintray_org}/#{bintray_repo}/#{filename}
    EOS
  end

  sig {
    params(bottles_hash:    T::Hash[String, T.untyped],
           publish_package: T::Boolean,
           warn_on_error:   T.nilable(T::Boolean)).void
  }
  def upload_bottles(bottles_hash, publish_package: false, warn_on_error: false)
    formula_packaged = {}

    bottles_hash.each do |formula_name, bottle_hash|
      version = ERB::Util.url_encode(bottle_hash["formula"]["pkg_version"])
      bintray_package = bottle_hash["bintray"]["package"]
      bintray_repo = bottle_hash["bintray"]["repository"]
      bottle_count = bottle_hash["bottle"]["tags"].length

      bottle_hash["bottle"]["tags"].each do |_tag, tag_hash|
        filename = tag_hash["filename"] # URL encoded in Bottle::Filename#bintray
        sha256 = tag_hash["sha256"]
        delete_instructions = file_delete_instructions(bintray_repo, bintray_package, filename)

        odebug "Checking remote file #{@bintray_org}/#{bintray_repo}/#{filename}"
        result = remote_checksum(repo: bintray_repo, remote_file: filename)

        case result
        when nil
          # File doesn't exist.
          if !formula_packaged[formula_name] && !package_exists?(repo: bintray_repo, package: bintray_package)
            odebug "Creating package #{@bintray_org}/#{bintray_repo}/#{bintray_package}"
            create_package repo: bintray_repo, package: bintray_package
            formula_packaged[formula_name] = true
          end

          odebug "Uploading #{@bintray_org}/#{bintray_repo}/#{bintray_package}/#{version}/#{filename}"
          upload(tag_hash["local_filename"],
                 repo:          bintray_repo,
                 package:       bintray_package,
                 version:       version,
                 remote_file:   filename,
                 sha256:        sha256,
                 warn_on_error: warn_on_error)
        when sha256
          # File exists, checksum matches.
          odebug "#{filename} is already published with matching hash."
          bottle_count -= 1
        when ""
          # File exists, but can't find checksum
          failed_message = "#{filename} is already published!"
          raise Error, "#{failed_message}\n#{delete_instructions}" unless warn_on_error

          opoo failed_message
        else
          # File exists, but checksum either doesn't exist or is mismatched.
          failed_message = <<~EOS
            #{filename} is already published with a mismatched hash!
              Expected: #{sha256}
              Actual:   #{result}
          EOS
          raise Error, "#{failed_message}#{delete_instructions}" unless warn_on_error

          opoo failed_message
        end
      end
      next unless publish_package

      odebug "Publishing #{@bintray_org}/#{bintray_repo}/#{bintray_package}/#{version}"
      publish(repo:          bintray_repo,
              package:       bintray_package,
              version:       version,
              file_count:    bottle_count,
              warn_on_error: warn_on_error)
    end
  end
end
