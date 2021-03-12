# typed: false
# frozen_string_literal: true

require "utils/curl"
require "json"

# GitHub Packages client.
#
# @api private
class GitHubPackages
  extend T::Sig

  include Context
  include Utils::Curl

  URL_REGEX = %r{https://ghcr.io/v2/([\w-]+)/([\w-]+)}.freeze

  sig { returns(String) }
  def inspect
    "#<GitHubPackages: org=#{@github_org}>"
  end

  sig { params(org: T.nilable(String)).void }
  def initialize(org: "homebrew")
    @github_org = org

    raise UsageError, "Must set a GitHub organisation!" unless @github_org

    ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"] = "1" if @github_org == "homebrew" && !OS.mac?
  end

  sig { params(bottles_hash: T::Hash[String, T.untyped]).void }
  def upload_bottles(bottles_hash)
    user = Homebrew::EnvConfig.github_packages_user
    token = Homebrew::EnvConfig.github_packages_token

    raise UsageError, "HOMEBREW_GITHUB_PACKAGES_USER is unset." if user.blank?
    raise UsageError, "HOMEBREW_GITHUB_PACKAGES_TOKEN is unset." if token.blank?

    oras   = Formula["oras"].opt_bin/"oras" if Formula["oras"].any_version_installed?
    oras ||= begin
      ohai "Installing `oras` for upload..."
      safe_system HOMEBREW_BREW_FILE, "install", "oras"
      Formula["oras"].opt_bin/"oras"
    end

    bottles_hash.each do |formula_name, bottle_hash|
      _, org, repo, = *bottle_hash["bottle"]["root_url"].match(URL_REGEX)
      version = bottle_hash["formula"]["pkg_version"]
      rebuild = bottle_hash["bottle"]["rebuild"]

      bottle_hash["bottle"]["tags"].each do |bottle_tag, tag_hash|
        local_file = tag_hash["local_filename"]
        odebug "Uploading #{local_file}"

        tag = "#{version}.#{bottle_tag}"
        tag = "#{tag}.#{rebuild}" if rebuild.positive?

        system_command!(oras, verbose: true, print_stdout: true, args: [
          "push", "ghcr.io/#{org}/#{repo}/#{formula_name}:#{tag}",
          "--username", user,
          "--password", token,
          "#{local_file}:application/vnd.oci.image.layer.v1.tar+gzip"
        ])

        # TODO: make this package public?
        # TODO: make this latest?
      end
    end
  end
end
