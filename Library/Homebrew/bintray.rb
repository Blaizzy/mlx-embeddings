# frozen_string_literal: true

require "utils/curl"
require "json"

# Bintray API client.
#
# @api private
class Bintray
  include Context

  API_URL = "https://api.bintray.com"

  class Error < RuntimeError
  end

  def inspect
    "#<Bintray: org=#{@bintray_org}>"
  end

  def initialize(org: "homebrew")
    @bintray_org = org

    raise UsageError, "Must set a Bintray organisation!" unless @bintray_org

    ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"] = "1" if @bintray_org == "homebrew" && !OS.mac?
  end

  def open_api(url, *extra_curl_args, auth: true)
    args = extra_curl_args

    if auth
      raise UsageError, "HOMEBREW_BINTRAY_USER is unset." unless (user = Homebrew::EnvConfig.bintray_user)
      raise UsageError, "HOMEBREW_BINTRAY_KEY is unset." unless (key = Homebrew::EnvConfig.bintray_key)

      args += ["--user", "#{user}:#{key}"]
    end

    curl(*args, url,
         print_stdout: false,
         secrets:      key)
  end

  def upload(local_file, repo:, package:, version:, remote_file:, sha256: nil)
    url = "#{API_URL}/content/#{@bintray_org}/#{repo}/#{package}/#{version}/#{remote_file}"
    args = ["--fail", "--upload-file", local_file]
    args += ["--header", "X-Checksum-Sha2: #{sha256}"] unless sha256.blank?
    result = open_api url, *args
    json = JSON.parse(result.stdout)
    raise "Bottle upload failed: #{json["message"]}" if json["message"] != "success"

    result
  end

  def publish(repo:, package:, version:, file_count:, warn_on_error: false)
    url = "#{API_URL}/content/#{@bintray_org}/#{repo}/#{package}/#{version}/publish"
    result = open_api url, "--request", "POST", "--fail"
    json = JSON.parse(result.stdout)
    if file_count.present? && json["files"] != file_count
      message = "Bottle publish failed: expected #{file_count} bottles, but published #{json["files"]} instead."
      raise message unless warn_on_error

      opoo message
    end

    odebug "Published #{json["files"]} bottles"
    result
  end

  def official_org?(org: @bintray_org)
    %w[homebrew linuxbrew].include? org
  end

  def stable_mirrored?(url)
    headers, = curl_output("--connect-timeout", "15", "--location", "--head", url)
    status_code = headers.scan(%r{^HTTP/.* (\d+)}).last.first
    status_code.start_with?("2")
  end

  def mirror_formula(formula, repo: "mirror", publish_package: false)
    package = Utils::Bottles::Bintray.package formula.name

    create_package(repo: repo, package: package) unless package_exists?(repo: repo, package: package)

    formula.downloader.fetch

    version = ERB::Util.url_encode(formula.pkg_version)
    filename = ERB::Util.url_encode(formula.downloader.basename)
    destination_url = "https://dl.bintray.com/#{@bintray_org}/#{repo}/#{filename}"

    odebug "Uploading to #{destination_url}"

    upload(
      formula.downloader.cached_location,
      repo:        repo,
      package:     package,
      version:     version,
      sha256:      formula.stable.checksum,
      remote_file: filename,
    )
    return destination_url unless publish_package

    odebug "Publishing #{@bintray_org}/#{repo}/#{package}/#{version}"
    publish(repo: repo, package: package, version: version, file_count: 1)

    destination_url
  end

  def create_package(repo:, package:, **extra_data_args)
    url = "#{API_URL}/packages/#{@bintray_org}/#{repo}"
    data = { name: package, public_download_numbers: true }
    data[:public_stats] = official_org?
    data.merge! extra_data_args
    open_api url, "--header", "Content-Type: application/json", "--request", "POST", "--data", data.to_json
  end

  def package_exists?(repo:, package:)
    url = "#{API_URL}/packages/#{@bintray_org}/#{repo}/#{package}"
    begin
      open_api url, "--fail", "--silent", "--output", "/dev/null", auth: false
    rescue ErrorDuringExecution => e
      stderr = e.output
                .select { |type,| type == :stderr }
                .map { |_, line| line }
                .join
      raise if e.status.exitstatus != 22 && !stderr.include?("404 Not Found")

      false
    else
      true
    end
  end

  def file_published?(repo:, remote_file:)
    url = "https://dl.bintray.com/#{@bintray_org}/#{repo}/#{remote_file}"
    begin
      curl "--fail", "--silent", "--head", "--output", "/dev/null", url
    rescue ErrorDuringExecution => e
      stderr = e.output
                .select { |type,| type == :stderr }
                .map { |_, line| line }
                .join
      raise if e.status.exitstatus != 22 && !stderr.include?("404 Not Found")

      false
    else
      true
    end
  end

  def upload_bottle_json(json_files, publish_package: false, warn_on_error: false)
    bottles_hash = json_files.reduce({}) do |hash, json_file|
      hash.deep_merge(JSON.parse(IO.read(json_file)))
    end

    formula_packaged = {}

    bottles_hash.each do |formula_name, bottle_hash|
      version = ERB::Util.url_encode(bottle_hash["formula"]["pkg_version"])
      bintray_package = bottle_hash["bintray"]["package"]
      bintray_repo = bottle_hash["bintray"]["repository"]

      bottle_hash["bottle"]["tags"].each do |_tag, tag_hash|
        filename = tag_hash["filename"] # URL encoded in Bottle::Filename#bintray
        sha256 = tag_hash["sha256"]

        odebug "Checking remote file #{@bintray_org}/#{bintray_repo}/#{filename}"
        if file_published? repo: bintray_repo, remote_file: filename
          already_published = "#{filename} is already published."
          failed_message = <<~EOS
            #{already_published}
            Please remove it manually from:
              https://bintray.com/#{@bintray_org}/#{bintray_repo}/#{bintray_package}/view#files
            Or run:
              curl -X DELETE -u $HOMEBREW_BINTRAY_USER:$HOMEBREW_BINTRAY_KEY \\
              https://api.bintray.com/content/#{@bintray_org}/#{bintray_repo}/#{filename}
          EOS
          raise Error, failed_message unless warn_on_error

          opoo already_published
          next
        end

        if !formula_packaged[formula_name] && !package_exists?(repo: bintray_repo, package: bintray_package)
          odebug "Creating package #{@bintray_org}/#{bintray_repo}/#{bintray_package}"
          create_package repo: bintray_repo, package: bintray_package
          formula_packaged[formula_name] = true
        end

        odebug "Uploading #{@bintray_org}/#{bintray_repo}/#{bintray_package}/#{version}/#{filename}"
        upload(tag_hash["local_filename"],
               repo:        bintray_repo,
               package:     bintray_package,
               version:     version,
               remote_file: filename,
               sha256:      sha256)
      end
      next unless publish_package

      bottle_count = bottle_hash["bottle"]["tags"].length
      odebug "Publishing #{@bintray_org}/#{bintray_repo}/#{bintray_package}/#{version}"
      publish(repo:          bintray_repo,
              package:       bintray_package,
              version:       version,
              file_count:    bottle_count,
              warn_on_error: warn_on_error)
    end
  end
end
