# typed: true
# frozen_string_literal: true

require "download_strategy"
require "utils/github"

module GitHub
  # Downloads an artifact from GitHub Actions.
  #
  # @param url [String] URL to download from
  # @param artifact_id [String] a value that uniquely identifies the downloaded artifact
  #
  # @api private
  sig { params(url: String, artifact_id: String).void }
  def self.download_artifact(url, artifact_id)
    odie "Credentials must be set to access the Artifacts API" if API.credentials_type == :none

    token = API.credentials

    # Download the artifact as a zip file and unpack it into `dir`. This is
    # preferred over system `curl` and `tar` as this leverages the Homebrew
    # cache to avoid repeated downloads of (possibly large) bottles.
    downloader = GitHubArtifactDownloadStrategy.new(url, artifact_id, token: token)
    downloader.fetch
    downloader.stage
  end
end

# Strategy for downloading an artifact from GitHub Actions.
#
# @api private
class GitHubArtifactDownloadStrategy < AbstractFileDownloadStrategy
  def initialize(url, artifact_id, token:)
    super(url, "artifact", artifact_id)
    @cache = HOMEBREW_CACHE/"gh-actions-artifact"
    @token = token
  end

  def fetch(timeout: nil)
    ohai "Downloading #{url}"
    if cached_location.exist?
      puts "Already downloaded: #{cached_location}"
    else
      begin
        curl "--location", "--create-dirs", "--output", temporary_path, url,
             "--header", "Authorization: token #{@token}",
             secrets: [@token],
             timeout: timeout
      rescue ErrorDuringExecution
        raise CurlDownloadStrategyError, url
      end
      ignore_interrupts do
        cached_location.dirname.mkpath
        temporary_path.rename(cached_location)
        symlink_location.dirname.mkpath
      end
    end
    FileUtils.ln_s cached_location.relative_path_from(symlink_location.dirname), symlink_location, force: true
  end

  private

  sig { returns(String) }
  def resolved_basename
    "artifact.zip"
  end
end
