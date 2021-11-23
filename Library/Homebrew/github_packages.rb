# typed: true
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
  DOCKER_PREFIX = "docker://#{URL_DOMAIN}/"
  public_constant :URL_DOMAIN
  private_constant :URL_PREFIX
  private_constant :DOCKER_PREFIX

  URL_REGEX = %r{(?:#{Regexp.escape(URL_PREFIX)}|#{Regexp.escape(DOCKER_PREFIX)})([\w-]+)/([\w-]+)}.freeze

  # Translate Homebrew tab.arch to OCI platform.architecture
  TAB_ARCH_TO_PLATFORM_ARCHITECTURE = {
    "arm64"  => "arm64",
    "x86_64" => "amd64",
  }.freeze

  # Translate Homebrew built_on.os to OCI platform.os
  BUILT_ON_OS_TO_PLATFORM_OS = {
    "Linux"     => "linux",
    "Macintosh" => "darwin",
  }.freeze

  sig {
    params(
      bottles_hash:  T::Hash[String, T.untyped],
      keep_old:      T::Boolean,
      dry_run:       T::Boolean,
      warn_on_error: T::Boolean,
    ).void
  }
  def upload_bottles(bottles_hash, keep_old:, dry_run:, warn_on_error:)
    user = Homebrew::EnvConfig.github_packages_user
    token = Homebrew::EnvConfig.github_packages_token

    raise UsageError, "HOMEBREW_GITHUB_PACKAGES_USER is unset." if user.blank?
    raise UsageError, "HOMEBREW_GITHUB_PACKAGES_TOKEN is unset." if token.blank?

    skopeo = ensure_executable!("skopeo", reason: "upload")

    require "json_schemer"

    load_schemas!

    bottles_hash.each do |formula_full_name, bottle_hash|
      upload_bottle(user, token, skopeo, formula_full_name, bottle_hash,
                    keep_old: keep_old, dry_run: dry_run, warn_on_error: warn_on_error)
    end
  end

  def self.version_rebuild(version, rebuild, bottle_tag = nil)
    bottle_tag = (".#{bottle_tag}" if bottle_tag.present?)

    rebuild = if rebuild.to_i.positive?
      if bottle_tag
        ".#{rebuild}"
      else
        "-#{rebuild}"
      end
    end

    "#{version}#{bottle_tag}#{rebuild}"
  end

  def self.repo_without_prefix(repo)
    # remove redundant repo prefix for a shorter name
    repo.delete_prefix("homebrew-")
  end

  def self.root_url(org, repo, prefix = URL_PREFIX)
    # docker/skopeo insist on lowercase org ("repository name")
    org = org.downcase

    "#{prefix}#{org}/#{repo_without_prefix(repo)}"
  end

  def self.root_url_if_match(url)
    return if url.blank?

    _, org, repo, = *url.to_s.match(URL_REGEX)
    return if org.blank? || repo.blank?

    root_url(org, repo)
  end

  def self.image_formula_name(formula_name)
    # invalid docker name characters
    # / makes sense because we already use it to separate repo/formula
    # x makes sense because we already use it in Formulary
    formula_name.tr("@", "/")
                .tr("+", "x")
  end

  def self.image_version_rebuild(version_rebuild)
    # invalid docker tag characters
    # TODO: consider changing the actual versions here and make an audit to
    # avoid these weird characters being used
    version_rebuild.gsub(/[+#~]/, ".")
  end

  private

  IMAGE_CONFIG_SCHEMA_URI = "https://opencontainers.org/schema/image/config"
  IMAGE_INDEX_SCHEMA_URI = "https://opencontainers.org/schema/image/index"
  IMAGE_LAYOUT_SCHEMA_URI = "https://opencontainers.org/schema/image/layout"
  IMAGE_MANIFEST_SCHEMA_URI = "https://opencontainers.org/schema/image/manifest"

  GITHUB_PACKAGE_TYPE = "homebrew_bottle"

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

  def download(user, token, skopeo, image_uri, root, dry_run:)
    puts
    args = ["copy", "--all", image_uri.to_s, "oci:#{root}"]
    if dry_run
      puts "#{skopeo} #{args.join(" ")} --src-creds=#{user}:$HOMEBREW_GITHUB_PACKAGES_TOKEN"
    else
      args << "--src-creds=#{user}:#{token}"
      system_command!(skopeo, verbose: true, print_stdout: true, args: args)
    end
  end

  def upload_bottle(user, token, skopeo, formula_full_name, bottle_hash, keep_old:, dry_run:, warn_on_error:)
    formula_name = bottle_hash["formula"]["name"]

    _, org, repo, = *bottle_hash["bottle"]["root_url"].match(URL_REGEX)
    repo = "homebrew-#{repo}" unless repo.start_with?("homebrew-")

    version = bottle_hash["formula"]["pkg_version"]
    rebuild = bottle_hash["bottle"]["rebuild"]
    version_rebuild = GitHubPackages.version_rebuild(version, rebuild)

    image_name = GitHubPackages.image_formula_name(formula_name)
    image_tag = GitHubPackages.image_version_rebuild(version_rebuild)
    image_uri = "#{GitHubPackages.root_url(org, repo, DOCKER_PREFIX)}/#{image_name}:#{image_tag}"

    puts
    inspect_args = ["inspect", "--raw", image_uri.to_s]
    if dry_run
      puts "#{skopeo} #{inspect_args.join(" ")} --creds=#{user}:$HOMEBREW_GITHUB_PACKAGES_TOKEN"
    else
      inspect_args << "--creds=#{user}:#{token}"
      inspect_result = system_command(skopeo, print_stderr: false, args: inspect_args)

      # Order here is important
      if !inspect_result.status.success? && !inspect_result.stderr.match?(/(name|manifest) unknown/)
        # We got an error, and it was not about the tag or package being unknown.
        if warn_on_error
          opoo "#{image_uri} inspection returned an error, skipping upload!\n#{inspect_result.stderr}"
          return
        else
          odie "#{image_uri} inspection returned an error!\n#{inspect_result.stderr}"
        end
      elsif keep_old
        # If the tag doesn't exist, ignore --keep-old.
        keep_old = false unless inspect_result.status.success?
        # Otherwise, do nothing - the tag already existing is expected behaviour for --keep-old.
      elsif inspect_result.status.success?
        # The tag already exists, and we are not passing --keep-old.
        if warn_on_error
          opoo "#{image_uri} already exists, skipping upload!"
          return
        else
          odie "#{image_uri} already exists!"
        end
      end
    end

    root = Pathname("#{formula_name}--#{version_rebuild}")
    FileUtils.rm_rf root
    root.mkpath

    if keep_old
      download(user, token, skopeo, image_uri, root, dry_run: dry_run)
    else
      write_image_layout(root)
    end

    blobs = root/"blobs/sha256"
    blobs.mkpath

    git_path = bottle_hash["formula"]["tap_git_path"]
    git_revision = bottle_hash["formula"]["tap_git_revision"]

    source_org_repo = "#{org}/#{repo}"
    source = "https://github.com/#{source_org_repo}/blob/#{git_revision.presence || "HEAD"}/#{git_path}"

    formula_core_tap = formula_full_name.exclude?("/")
    documentation = if formula_core_tap
      "https://formulae.brew.sh/formula/#{formula_name}"
    elsif (remote = bottle_hash["formula"]["tap_git_remote"]) && remote.start_with?("https://github.com/")
      remote
    end

    created_date = bottle_hash["bottle"]["date"]
    if keep_old
      index = JSON.parse((root/"index.json").read)
      image_index_sha256 = index["manifests"].first["digest"].delete_prefix("sha256:")
      image_index = JSON.parse((blobs/image_index_sha256).read)
      (blobs/image_index_sha256).unlink

      formula_annotations_hash = image_index["annotations"]
      manifests = image_index["manifests"]
    else
      formula_annotations_hash = {
        "com.github.package.type"                => GITHUB_PACKAGE_TYPE,
        "org.opencontainers.image.created"       => created_date,
        "org.opencontainers.image.description"   => bottle_hash["formula"]["desc"],
        "org.opencontainers.image.documentation" => documentation,
        "org.opencontainers.image.license"       => bottle_hash["formula"]["license"],
        "org.opencontainers.image.ref.name"      => version_rebuild,
        "org.opencontainers.image.revision"      => git_revision,
        "org.opencontainers.image.source"        => source,
        "org.opencontainers.image.title"         => formula_full_name,
        "org.opencontainers.image.url"           => bottle_hash["formula"]["homepage"],
        "org.opencontainers.image.vendor"        => org,
        "org.opencontainers.image.version"       => version,
      }.reject { |_, v| v.blank? }
      manifests = []
    end

    processed_image_refs = Set.new
    manifests.each do |manifest|
      processed_image_refs << manifest["annotations"]["org.opencontainers.image.ref.name"]
    end

    manifests += bottle_hash["bottle"]["tags"].map do |bottle_tag, tag_hash|
      bottle_tag = Utils::Bottles::Tag.from_symbol(bottle_tag.to_sym)

      tag = GitHubPackages.version_rebuild(version, rebuild, bottle_tag.to_s)

      if processed_image_refs.include?(tag)
        puts
        odie "A bottle JSON for #{bottle_tag} is present, but it is already in the image index!"
      else
        processed_image_refs << tag
      end

      local_file = tag_hash["local_filename"]
      odebug "Uploading #{local_file}"

      tar_gz_sha256 = write_tar_gz(local_file, blobs)

      tab = tag_hash["tab"]
      architecture = TAB_ARCH_TO_PLATFORM_ARCHITECTURE[tab["arch"].presence || bottle_tag.arch.to_s]
      raise TypeError, "unknown tab['arch']: #{tab["arch"]}" if architecture.blank?

      os = if tab["built_on"].present? && tab["built_on"]["os"].present?
        BUILT_ON_OS_TO_PLATFORM_OS[tab["built_on"]["os"]]
      elsif bottle_tag.linux?
        "linux"
      else
        "darwin"
      end
      raise TypeError, "unknown tab['built_on']['os']: #{tab["built_on"]["os"]}" if os.blank?

      os_version = tab["built_on"]["os_version"].presence if tab["built_on"].present?
      case os
      when "darwin"
        os_version ||= "macOS #{bottle_tag.to_macos_version}"
      when "linux"
        os_version&.delete_suffix!(" LTS")
        os_version ||= OS::CI_OS_VERSION
        glibc_version = tab["built_on"]["glibc_version"].presence if tab["built_on"].present?
        glibc_version ||= OS::CI_GLIBC_VERSION
        cpu_variant = tab["oldest_cpu_family"] || Hardware::CPU::INTEL_64BIT_OLDEST_CPU.to_s
      end

      platform_hash = {
        architecture: architecture,
        os: os,
        "os.version" => os_version,
      }.reject { |_, v| v.blank? }

      tar_sha256 = Digest::SHA256.hexdigest(
        Utils.safe_popen_read("gunzip", "--stdout", "--decompress", local_file),
      )

      config_json_sha256, config_json_size = write_image_config(platform_hash, tar_sha256, blobs)

      formulae_dir = tag_hash["formulae_brew_sh_path"]
      documentation = "https://formulae.brew.sh/#{formulae_dir}/#{formula_name}" if formula_core_tap

      descriptor_annotations_hash = {
        "org.opencontainers.image.ref.name" => tag,
        "sh.brew.bottle.cpu.variant"        => cpu_variant,
        "sh.brew.bottle.digest"             => tar_gz_sha256,
        "sh.brew.bottle.glibc.version"      => glibc_version,
        "sh.brew.tab"                       => tab.to_json,
      }.reject { |_, v| v.blank? }

      annotations_hash = formula_annotations_hash.merge(descriptor_annotations_hash).merge(
        {
          "org.opencontainers.image.created"       => created_date,
          "org.opencontainers.image.documentation" => documentation,
          "org.opencontainers.image.title"         => "#{formula_full_name} #{tag}",
        },
      ).reject { |_, v| v.blank? }.sort.to_h

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
        annotations: descriptor_annotations_hash,
      }
    end

    index_json_sha256, index_json_size = write_image_index(manifests, blobs, formula_annotations_hash)

    write_index_json(index_json_sha256, index_json_size, root,
                     "org.opencontainers.image.ref.name" => version_rebuild)

    puts
    args = ["copy", "--all", "oci:#{root}", image_uri.to_s]
    if dry_run
      puts "#{skopeo} #{args.join(" ")} --dest-creds=#{user}:$HOMEBREW_GITHUB_PACKAGES_TOKEN"
    else
      args << "--dest-creds=#{user}:#{token}"
      system_command!(skopeo, verbose: true, print_stdout: true, args: args)
      package_name = "#{GitHubPackages.repo_without_prefix(repo)}/#{image_name}"
      ohai "Uploaded to https://github.com/orgs/#{org}/packages/container/package/#{package_name}"
    end
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
      schemaVersion: 2,
      manifests:     manifests,
      annotations:   annotations,
    }
    validate_schema!(IMAGE_INDEX_SCHEMA_URI, image_index)
    write_hash(blobs, image_index)
  end

  def write_index_json(index_json_sha256, index_json_size, root, annotations)
    index_json = {
      schemaVersion: 2,
      manifests:     [{
        mediaType:   "application/vnd.oci.image.index.v1+json",
        digest:      "sha256:#{index_json_sha256}",
        size:        index_json_size,
        annotations: annotations,
      }],
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
