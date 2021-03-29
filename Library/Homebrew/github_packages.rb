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

    # TODO: these dependencies are installed but cannot be required automatically.
    Homebrew.install_gem!("json_schemer")
    require "json_schemer"

    load_schemas!

    bottles_hash.each_value do |bottle_hash|
      upload_bottle(user, token, skopeo, bottle_hash)
    end
  end

  private

  IMAGE_CONFIG_SCHEMA_URI = "https://opencontainers.org/schema/image/config"
  IMAGE_INDEX_SCHEMA_URI = "https://opencontainers.org/schema/image/index"
  IMAGE_LAYOUT_SCHEMA_URI = "https://opencontainers.org/schema/image/layout"
  IMAGE_MANIFEST_SCHEMA_URI = "https://opencontainers.org/schema/image/manifest"

  def load_schemas!
    schema_uri("content-descriptor",
               "https://opencontainers.org/schema/image/content-descriptor.json")
    schema_uri("defs", %w[
      https://opencontainers.org/schema/defs.json
      https://opencontainers.org/schema/descriptor/defs.json
      https://opencontainers.org/schema/image/defs.json
      https://opencontainers.org/schema/image/descriptor/defs.json
      https://opencontainers.org/schema/image/index/defs.json
      https://opencontainers.org/schema/image/manifest/defs.json
    ])
    schema_uri("defs-descriptor", %w[
      https://opencontainers.org/schema/descriptor.json
      https://opencontainers.org/schema/defs-descriptor.json
      https://opencontainers.org/schema/descriptor/defs-descriptor.json
      https://opencontainers.org/schema/image/defs-descriptor.json
      https://opencontainers.org/schema/image/descriptor/defs-descriptor.json
      https://opencontainers.org/schema/image/index/defs-descriptor.json
      https://opencontainers.org/schema/image/manifest/defs-descriptor.json
      https://opencontainers.org/schema/index/defs-descriptor.json
    ])
    schema_uri("config-schema", IMAGE_CONFIG_SCHEMA_URI)
    schema_uri("image-index-schema", IMAGE_INDEX_SCHEMA_URI)
    schema_uri("image-layout-schema", IMAGE_LAYOUT_SCHEMA_URI)
    schema_uri("image-manifest-schema", IMAGE_MANIFEST_SCHEMA_URI)
  end

  def schema_uri(basename, uris)
    url = "https://raw.githubusercontent.com/opencontainers/image-spec/master/schema/#{basename}.json"
    out, = curl_output(url)
    json = JSON.parse(out)

    @schema_json ||= {}
    Array(uris).each do |uri|
      @schema_json[uri] = json
    end
  end

  def schema_resolver(uri)
    @schema_json[uri.to_s.gsub(/#.*/, "")]
  end

  def validate_schema!(schema_uri, json)
    schema = JSONSchemer.schema(@schema_json[schema_uri], ref_resolver: method(:schema_resolver))
    json = json.deep_stringify_keys
    return if schema.valid?(json)

    puts
    ofail "#{Formatter.url(schema_uri)} JSON schema validation failed!"
    oh1 "Errors"
    pp schema.validate(json).to_a
    oh1 "JSON"
    pp json
    exit 1
  end

  def upload_bottle(user, token, skopeo, bottle_hash)
    formula_path = HOMEBREW_REPOSITORY/bottle_hash["formula"]["path"]
    formula = Formulary.factory(formula_path)
    formula_name = formula.name

    _, org, repo, = *bottle_hash["bottle"]["root_url"].match(URL_REGEX)

    # docker/skopeo insist on lowercase org ("repository name")
    org = org.downcase

    version = bottle_hash["formula"]["pkg_version"]
    rebuild = if (rebuild = bottle_hash["bottle"]["rebuild"]).positive?
      ".#{rebuild}"
    end
    version_rebuild = "#{version}#{rebuild}"
    root = Pathname("#{formula_name}-#{version_rebuild}")
    FileUtils.rm_rf root

    write_image_layout(root)

    blobs = root/"blobs/sha256"
    blobs.mkpath

    # TODO: ideally most/all of these attributes would be stored in the
    # bottle JSON rather than reading them from the formula.
    git_revision = formula.tap.git_head
    git_path = formula_path.to_s.delete_prefix("#{formula.tap.path}/")
    source = "https://github.com/#{org}/#{repo}/blob/#{git_revision}/#{git_path}"

    formula_annotations_hash = {
      "org.opencontainers.image.created"     => Time.now.strftime("%F"),
      "org.opencontainers.image.description" => formula.desc,
      "org.opencontainers.image.license"     => formula.license,
      "org.opencontainers.image.revision"    => git_revision,
      "org.opencontainers.image.source"      => source,
      "org.opencontainers.image.url"         => formula.homepage,
      "org.opencontainers.image.vendor"      => org,
      "org.opencontainers.image.version"     => version,
    }
    formula_annotations_hash.each do |key, value|
      formula_annotations_hash.delete(key) if value.blank?
    end

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
      os, arch, formulae_dir = if bottle_tag.to_s.end_with?("_linux")
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

      config_json_sha256, config_json_size = write_image_config(platform_hash, tar_sha256, blobs)

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

      image_manifest = {
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
            "org.opencontainers.image.title" => local_file,
          },
        }],
        annotations:   annotations_hash,
      }
      validate_schema!(IMAGE_MANIFEST_SCHEMA_URI, image_manifest)
      manifest_json_sha256, manifest_json_size = write_hash(blobs, image_manifest)

      {
        mediaType:   "application/vnd.oci.image.manifest.v1+json",
        digest:      "sha256:#{manifest_json_sha256}",
        size:        manifest_json_size,
        platform:    platform_hash,
        annotations: {
          "org.opencontainers.image.ref.name" => tag,
        },
      }
    end

    index_json_sha256, index_json_size = write_image_index(manifests, blobs, formula_annotations_hash)

    write_index_json(index_json_sha256, index_json_size, root)

    image = "#{URL_DOMAIN}/#{org}/#{repo}/#{formula_name}"
    image_tag = "#{image}:#{version_rebuild}"
    puts
    system_command!(skopeo, verbose: true, print_stdout: true, args: [
      "copy", "--dest-creds=#{user}:#{token}",
      "oci:#{root}", "docker://#{image_tag}"
    ])
  end

  def write_image_layout(root)
    image_layout = { imageLayoutVersion: "1.0.0" }
    validate_schema!(IMAGE_LAYOUT_SCHEMA_URI, image_layout)
    write_hash(root, image_layout, "oci-layout")
  end

  def write_tar_gz(local_file, blobs)
    tar_gz_sha256 = Digest::SHA256.file(local_file)
                                  .hexdigest
    FileUtils.cp local_file, blobs/tar_gz_sha256
    tar_gz_sha256
  end

  def write_image_config(platform_hash, tar_sha256, blobs)
    image_config = platform_hash.merge({
      rootfs: {
        type:     "layers",
        diff_ids: ["sha256:#{tar_sha256}"],
      },
    })
    validate_schema!(IMAGE_CONFIG_SCHEMA_URI, image_config)
    write_hash(blobs, image_config)
  end

  def write_image_index(manifests, blobs, annotations)
    image_index = {
      # Currently needed for correct multi-arch display in GitHub Packages UI
      mediaType:     "application/vnd.docker.distribution.manifest.list.v2+json",
      schemaVersion: 2,
      manifests:     manifests,
      annotations:   annotations,
    }
    validate_schema!(IMAGE_INDEX_SCHEMA_URI, image_index)
    write_hash(blobs, image_index)
  end

  def write_index_json(index_json_sha256, index_json_size, root)
    index_json = {
      schemaVersion: 2,
      manifests:     [{
        mediaType:   "application/vnd.oci.image.index.v1+json",
        digest:      "sha256:#{index_json_sha256}",
        size:        index_json_size,
        annotations: {},
      }],
      annotations:   {},
    }
    validate_schema!(IMAGE_INDEX_SCHEMA_URI, index_json)
    write_hash(root, index_json, "index.json")
  end

  def write_hash(directory, hash, filename = nil)
    json = JSON.pretty_generate(hash)
    sha256 = Digest::SHA256.hexdigest(json)
    filename ||= sha256
    path = directory/filename
    path.unlink if path.exist?
    path.write(json)

    [sha256, json.size]
  end
end
