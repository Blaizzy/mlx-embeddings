# typed: true
# frozen_string_literal: true

require "github_packages"

# Helper functions for using the Bottle JSON API.
#
# @api private
module BottleAPI
  extend T::Sig

  module_function

  FORMULAE_BREW_SH_BOTTLE_API_DOMAIN = if OS.mac?
    "https://formulae.brew.sh/api/bottle"
  else
    "https://formulae.brew.sh/api/bottle-linux"
  end.freeze

  FORMULAE_BREW_SH_VERSIONS_API_URL = if OS.mac?
    "https://formulae.brew.sh/api/versions-formulae.json"
  else
    "https://formulae.brew.sh/api/versions-linux.json"
  end.freeze

  GITHUB_PACKAGES_SHA256_REGEX = %r{#{GitHubPackages::URL_REGEX}.*/blobs/sha256:(?<sha256>\h{64})$}.freeze

  sig { params(name: String).returns(Hash) }
  def fetch(name)
    return @cache[name] if @cache.present? && @cache.key?(name)

    api_url = "#{FORMULAE_BREW_SH_BOTTLE_API_DOMAIN}/#{name}.json"
    output = Utils::Curl.curl_output("--fail", api_url)
    raise ArgumentError, "No JSON file found at #{Tty.underline}#{api_url}#{Tty.reset}" unless output.success?

    @cache ||= {}
    @cache[name] = JSON.parse(output.stdout)
  rescue JSON::ParserError
    raise ArgumentError, "Invalid JSON file: #{Tty.underline}#{api_url}#{Tty.reset}"
  end

  sig { params(name: String).returns(T.nilable(PkgVersion)) }
  def latest_pkg_version(name)
    @formula_versions ||= begin
      output = Utils::Curl.curl_output("--fail", FORMULAE_BREW_SH_VERSIONS_API_URL)
      JSON.parse(output.stdout)
    end

    return unless @formula_versions.key? name

    version = Version.new(@formula_versions[name]["version"])
    revision = @formula_versions[name]["revision"]
    PkgVersion.new(version, revision)
  end

  sig { params(name: String).returns(T::Boolean) }
  def bottle_available?(name)
    fetch name
    true
  rescue ArgumentError
    false
  end

  sig { params(name: String).void }
  def fetch_bottles(name)
    hash = fetch(name)
    bottle_tag = Utils::Bottles.tag.to_s

    odie "No bottle available for current OS" unless hash["bottles"].key? bottle_tag

    download_bottle(hash, bottle_tag)

    hash["dependencies"].each do |dep_hash|
      download_bottle(dep_hash, bottle_tag)
    end
  end

  sig { params(url: String).returns(T.nilable(String)) }
  def checksum_from_url(url)
    match = url.match GITHUB_PACKAGES_SHA256_REGEX
    return if match.blank?

    match[:sha256]
  end

  sig { params(hash: Hash, tag: Symbol).void }
  def download_bottle(hash, tag)
    bottle = hash["bottles"][tag]
    return if bottle.blank?

    sha256 = bottle["sha256"] || checksum_from_url(bottle["url"])
    bottle_filename = Bottle::Filename.new(hash["name"], hash["pkg_version"], tag, hash["rebuild"])

    resource = Resource.new hash["name"]
    resource.url bottle["url"]
    resource.sha256 sha256
    resource.version hash["pkg_version"]
    resource.downloader.resolved_basename = bottle_filename

    resource.fetch

    # Map the name of this formula to the local bottle path to allow the
    # formula to be loaded by passing just the name to `Formulary::factory`.
    Formulary.map_formula_name_to_local_bottle_path hash["name"], resource.downloader.cached_location
  end
end
