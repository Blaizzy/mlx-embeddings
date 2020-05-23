# frozen_string_literal: true

require "utils/curl"
require "json"

class Bintray
  API_URL = "https://api.bintray.com"

  class Error < RuntimeError
  end

  def inspect
    "#<Bintray: user=#{@bintray_user} org=#{@bintray_org} key=***>"
  end

  def initialize(user: ENV["HOMEBREW_BINTRAY_USER"], key: ENV["HOMEBREW_BINTRAY_KEY"], org: "homebrew", clear: true)
    @bintray_user = user
    @bintray_key = key
    @bintray_org = org

    if !@bintray_user || !@bintray_key
      unless Homebrew.args.dry_run?
        raise UsageError, "Missing HOMEBREW_BINTRAY_USER or HOMEBREW_BINTRAY_KEY variables!"
      end
    end

    raise UsageError, "Must set a Bintray organisation!" unless @bintray_org

    ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"] = "1" if @bintray_org == "homebrew" && !OS.mac?

    ENV.delete "HOMEBREW_BINTRAY_KEY" if clear
  end

  def open_api(url, *extra_curl_args, auth: true)
    args = extra_curl_args
    args += ["--user", "#{@bintray_user}:#{@bintray_key}"] if auth
    args += ["--output", "/dev/null"] unless Homebrew.args.verbose?
    curl(*args, url,
         show_output: Homebrew.args.verbose?,
         secrets:     @bintray_key)
  end

  def upload(local_file, repo:, package:, version:, remote_file:, sha256: nil)
    url = "#{API_URL}/content/#{@bintray_org}/#{repo}/#{package}/#{version}/#{remote_file}"
    args = ["--upload-file", local_file]
    args += ["--header", "X-Checksum-Sha2: #{sha256}"] unless sha256.blank?
    open_api url, *args
  end

  def publish(repo:, package:, version:)
    url = "#{API_URL}/content/#{@bintray_org}/#{repo}/#{package}/#{version}/publish"
    open_api url, "--request", "POST"
  end

  def official_org?(org: @bintray_org)
    %w[homebrew linuxbrew].include? org
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
      open_api url, "--fail", "--silent", auth: false
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

  def upload_bottle_json(json_files, publish_package: false)
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
          raise Error, <<~EOS
            #{filename} is already published.
            Please remove it manually from:
              https://bintray.com/#{@bintray_org}/#{bintray_repo}/#{bintray_package}/view#files
            Or run:
              curl -X DELETE -u $HOMEBREW_BINTRAY_USER:$HOMEBREW_BINTRAY_KEY \\
              https://api.bintray.com/content/#{@bintray_org}/#{bintray_repo}/#{filename}
          EOS
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
      if publish_package
        odebug "Publishing #{@bintray_org}/#{bintray_repo}/#{bintray_package}/#{version}"
        publish repo: bintray_repo, package: bintray_package, version: version
      end
    end
  end
end
