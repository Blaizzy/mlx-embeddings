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

    skopeo = HOMEBREW_PREFIX/"bin/skopeo"
    unless skopeo.exist?
      ohai "Installing `skopeo` for upload..."
      safe_system HOMEBREW_BREW_FILE, "install", "--formula", "skopeo"
      skopeo = Formula["skopeo"].opt_bin/"skopeo"
    end

    bottles_hash.each do |formula_name, bottle_hash|
      upload_bottle(user, token, skopeo, formula_name, bottle_hash)
    end
  end

  private

  def upload_bottle(user, token, skopeo, formula_name, bottle_hash)
    _, org, repo, = *bottle_hash["bottle"]["root_url"].match(URL_REGEX)

    # docker/skopeo insist on lowercase org ("repository name")
    org = org.downcase

    version = bottle_hash["formula"]["pkg_version"]
    rebuild = if (rebuild = bottle_hash["bottle"]["rebuild"]).positive?
      ".#{rebuild}"
    end
    version_rebuild = "#{version}#{rebuild}"
    root = Pathname("#{formula_name}-#{version_rebuild}")

    write_oci_layout(root)

    blobs = root/"blobs/sha256"
    blobs.mkpath

    formula_path = HOMEBREW_REPOSITORY/bottle_hash["formula"]["path"]
    formula = Formulary.factory(formula_path)

    # TODO: ideally most/all of these attributes would be stored in the
    # bottle JSON rather than reading them from the formula.
    git_revision = formula.tap.git_head
    git_path = formula_path.to_s.delete_prefix("#{formula.tap.path}/")
    source = "https://github.com/#{org}/#{repo}/blob/#{git_revision}/#{git_path}"

    formula_annotations_hash = {
      "org.opencontainers.image.description" => formula.desc,
      "org.opencontainers.image.license"     => formula.license,
      "org.opencontainers.image.revision"    => git_revision,
      "org.opencontainers.image.source"      => source,
      "org.opencontainers.image.url"         => formula.homepage,
      "org.opencontainers.image.vendor"      => org,
      "org.opencontainers.image.version"     => version,
    }

    manifests = bottle_hash["bottle"]["tags"].map do |bottle_tag, tag_hash|
      local_file = tag_hash["local_filename"]
      odebug "Uploading #{local_file}"

      tar_gz_sha256 = write_tar_gz(local_file, blobs)

      tab = Tab.from_file_content(
        Utils.safe_popen_read("tar", "xfO", local_file, "#{formula_name}/#{version}/INSTALL_RECEIPT.json"),
        "#{local_file}/#{formula_name}/#{version}",
      )
      os_version = if tab.built_on.present?
        /(\d+\.)*\d+/ =~ tab.built_on["os_version"]
        Regexp.last_match(0)
      end

      # TODO: ideally most/all of these attributes would be stored in the
      # bottle JSON rather than reading them from the formula.
      os, arch, formulae_dir = if @bottle_tag.to_s.end_with?("_linux")
        ["linux", "amd64", "formula-linux"]
      else
        os = "darwin"
        macos_version = MacOS::Version.from_symbol(bottle_tag.to_sym)
        os_version ||= macos_version.to_f.to_s
        arch = if macos_version.arch == :arm64
          "arm64"
        else
          "amd64"
        end
        [os, arch, "formula"]
      end

      platform_hash = {
        architecture: arch,
        os: os,
        "os.version" => os_version,
      }
      tar_sha256 = Digest::SHA256.hexdigest(
        Utils.safe_popen_read("gunzip", "--stdout", "--decompress", local_file),
      )

      config_json_sha256, config_json_size = write_config(platform_hash, tar_sha256, blobs)

      created_time = tab.source_modified_time
      created_time ||= Time.now
      documentation = "https://formulae.brew.sh/#{formulae_dir}/#{formula_name}" if formula.tap.core_tap?
      tag = "#{version}.#{bottle_tag}#{rebuild}"
      title = "#{formula.full_name} #{tag}"

      annotations_hash = formula_annotations_hash.merge({
        "org.opencontainers.image.created"       => created_time.strftime("%F"),
        "org.opencontainers.image.documentation" => documentation,
        "org.opencontainers.image.ref.name"      => tag,
        "org.opencontainers.image.title"         => title,
      }).sort.to_h
      annotations_hash.each do |key, value|
        annotations_hash.delete(key) if value.blank?
      end

      manifest_json_sha256, manifest_json_size = write_hash(blobs, {
        schemaVersion: 2,
        config:        {
          mediaType: "application/vnd.oci.image.config.v1+json",
          digest:    "sha256:#{config_json_sha256}",
          size:      config_json_size,
        },
        layers:        [{
          mediaType:   "application/vnd.oci.image.layer.v1.tar+gzip",
          digest:      "sha256:#{tar_gz_sha256}",
          size:        File.size(local_file),
          annotations: {
            "org.opencontainers.image.title": local_file,
          },
        }],
        annotations:   annotations_hash,
      })

      {
        mediaType: "application/vnd.oci.image.manifest.v1+json",
        digest:    "sha256:#{manifest_json_sha256}",
        size:      manifest_json_size,
        platform:  platform_hash,
      }
    end

    index_json_sha256, index_json_size = write_index(manifests, blobs)

    write_index_json(index_json_sha256, index_json_size, root)

    image = "#{URL_DOMAIN}/#{org}/#{repo}/#{formula_name}"
    image_tag = "#{image}:#{version_rebuild}"
    puts
    system_command!(skopeo, verbose: true, print_stdout: true, args: [
      "copy", "--dest-creds=#{user}:#{token}",
      "oci:#{root}", "docker://#{image_tag}"
    ])
  end

  def write_oci_layout(root)
    write_hash(root, { imageLayoutVersion: "1.0.0" }, "oci-layout")
  end

  def write_tar_gz(local_file, blobs)
    tar_gz_sha256 = Digest::SHA256.file(local_file)
                                  .hexdigest
    FileUtils.cp local_file, blobs/tar_gz_sha256
    tar_gz_sha256
  end

  def write_config(platform_hash, tar_sha256, blobs)
    write_hash(blobs, platform_hash.merge({
      rootfs: {
        type:     "layers",
        diff_ids: ["sha256:#{tar_sha256}"],
      },
    }))
  end

  def write_index(manifests, blobs)
    write_hash(blobs, {
      schemaVersion: 2,
      manifests:     manifests,
    })
  end

  def write_index_json(index_json_sha256, index_json_size, root)
    write_hash(root, {
      schemaVersion: 2,
      manifests:     [{
        mediaType: "application/vnd.oci.image.index.v1+json",
        digest:    "sha256:#{index_json_sha256}",
        size:      index_json_size,
      }],
    }, "index.json")
  end

  def write_hash(directory, hash, _filename = nil)
    json = hash.to_json
    sha256 = Digest::SHA256.hexdigest(json)
    path = directory/sha256
    path.unlink if path.exist?
    path.write(json)

    [sha256, json.size]
  end
end
