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

  URL_DOMAIN = "ghcr.io"
  URL_PREFIX = "https://#{URL_DOMAIN}/v2/"
  URL_REGEX = %r{#{Regexp.escape(URL_PREFIX)}([\w-]+)/([\w-]+)}.freeze

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

    docker = HOMEBREW_PREFIX/"bin/docker"
    unless docker.exist?
      ohai "Installing `docker` for upload..."
      safe_system HOMEBREW_BREW_FILE, "install", "--formula", "docker"
      docker = Formula["docker"].opt_bin/"docker"
    end

    puts
    system_command!(docker, verbose: true, print_stdout: true, input: token, args: [
      "login", "--username", user, "--password-stdin", URL_DOMAIN
    ])

    oras = HOMEBREW_PREFIX/"bin/oras"
    unless oras.exist?
      ohai "Installing `oras` for upload..."
      safe_system HOMEBREW_BREW_FILE, "install", "oras"
      oras = Formula["oras"].opt_bin/"oras"
    end

    bottles_hash.each do |formula_name, bottle_hash|
      _, org, repo, = *bottle_hash["bottle"]["root_url"].match(URL_REGEX)

      # docker CLI insists on lowercase org ("repository name")
      org = org.downcase
      image = "#{URL_DOMAIN}/#{org}/#{repo}/#{formula_name}"

      version = bottle_hash["formula"]["pkg_version"]
      rebuild = if (rebuild = bottle_hash["bottle"]["rebuild"]).positive?
        ".#{rebuild}"
      end

      formula_path = HOMEBREW_REPOSITORY/bottle_hash["formula"]["path"]
      formula = Formulary.factory(formula_path)

      image_tags = bottle_hash["bottle"]["tags"].map do |bottle_tag, tag_hash|
        local_file = tag_hash["local_filename"]
        odebug "Uploading #{local_file}"

        tag = "#{version}.#{bottle_tag}#{rebuild}"

        tab = Tab.from_file_content(
          Utils.safe_popen_read("tar", "xfO", local_file, "#{formula_name}/#{version}/INSTALL_RECEIPT.json"),
          "#{local_file}/#{formula_name}/#{version}",
        )
        created_time = tab.source_modified_time
        created_time ||= Time.now

        # TODO: ideally most/all of these attributes would be stored in the
        # bottle JSON rather than reading them from the formula.
        git_revision = formula.tap.git_head
        git_path = formula_path.to_s.delete_prefix("#{formula.tap.path}/")
        manifest_hash = {
          "org.opencontainers.image.title"    => formula.full_name,
          "org.opencontainers.image.url"      => formula.homepage,
          "org.opencontainers.image.version"  => version,
          "org.opencontainers.image.revision" => git_revision,
          "org.opencontainers.image.source"   => "https://github.com/#{org}/#{repo}/blob/#{git_revision}/#{git_path}",
          "org.opencontainers.image.created"  => created_time.strftime("%F"),
        }
        manifest_hash["org.opencontainers.image.description"] = formula.desc if formula.desc.present?
        manifest_hash["org.opencontainers.image.license"] = formula.license if formula.license.present?

        manifest_annotations = Pathname("#{formula_name}.#{tag}.annotations.json")
        manifest_annotations.unlink if manifest_annotations.exist?
        manifest_annotations.write({ "$manifest" => manifest_hash }.to_json)

        os_version = if tab.built_on.present?
          /(\d+\.)*\d+/ =~ tab.built_on["os_version"]
          Regexp.last_match(0)
        end

        # TODO: ideally most/all of these attributes would be stored in the
        # bottle JSON rather than reading them from the formula.
        os, arch = if @bottle_tag.to_s.end_with?("_linux")
          ["linux", "amd64"]
        else
          os = "darwin"
          macos_version = MacOS::Version.from_symbol(bottle_tag.to_sym)
          os_version ||= macos_version.to_f.to_s
          arch = if macos_version.arch == :arm64
            "arm64"
          else
            "amd64"
          end
          [os, arch]
        end

        tar_sha256 = Digest::SHA256.hexdigest(
          Utils.safe_popen_read("gunzip", "--stdout", "--decompress", local_file),
        )

        config_hash = {
          "architecture" => arch,
          "os"           => os,
          "os.version"   => os_version,
          "rootfs"       => {
            "type"     => "layers",
            "diff_ids" => ["sha256:#{tar_sha256}"],
          },
        }

        manifest_config = Pathname("#{formula_name}.#{tag}.config.json")
        manifest_config.unlink if manifest_config.exist?
        manifest_config.write(config_hash.to_json)

        # TODO: If we push the architecture-specific images to the tag :latest,
        # then we don't need to delete the architecture-specific tags.
        image_tag = "#{image}:#{tag}"
        puts
        system_command!(oras, verbose: true, print_stdout: true, args: [
          "push", image_tag,
          "--verbose",
          "--manifest-annotations=#{manifest_annotations}",
          "--manifest-config=#{manifest_config}:application/vnd.oci.image.config.v1+json",
          "--username", user,
          "--password", token,
          "#{local_file}:application/vnd.oci.image.layer.v1.tar+gzip"
        ])

        image_tag
      end

      image_tag = "#{image}:#{version}#{rebuild}"
      puts
      system_command!(docker, verbose: true, print_stdout: true, args: [
        "buildx", "imagetools", "create", "--tag", image_tag, *image_tags
      ])

      # TODO: once the main image metadata is working correctly delete the package using:
      # `curl -X DELETE -u $HOMEBREW_GITHUB_PACKAGES_USER:$HOMEBREW_GITHUB_PACKAGES_TOKEN
      #  https://api.github.com/orgs/Homebrew/packages/container/homebrew-core%2F$PACKAGE/versions/$VERSION`
      # Alternatively, if we push the architecture-specific images to the tag :latest,
      # then we don't need to delete the architecture-specific tags.
      # Alternatively, remove all usage of `docker` here instead.
    end
  ensure
    if docker
      puts
      system_command!(docker, verbose: true, print_stdout: true, args: [
        "logout", URL_DOMAIN
      ])
    end
  end
end
