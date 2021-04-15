# typed: true
# frozen_string_literal: true

require "download_strategy"
require "checksum"
require "version"
require "mktemp"
require "extend/on_os"

# Resource is the fundamental representation of an external resource. The
# primary formula download, along with other declared resources, are instances
# of this class.
#
# @api private
class Resource
  extend T::Sig

  include Context
  include FileUtils
  include OnOS

  attr_reader :mirrors, :specs, :using, :source_modified_time, :patches, :owner
  attr_writer :version
  attr_accessor :download_strategy, :checksum

  # Formula name must be set after the DSL, as we have no access to the
  # formula name before initialization of the formula.
  attr_accessor :name

  def initialize(name = nil, &block)
    @name = name
    @url = nil
    @version = nil
    @mirrors = []
    @specs = {}
    @checksum = nil
    @using = nil
    @patches = []
    instance_eval(&block) if block
  end

  def owner=(owner)
    @owner = owner
    patches.each { |p| p.owner = owner }
  end

  def downloader
    @downloader ||= download_strategy.new(url, download_name, version,
                                          mirrors: mirrors.dup, **specs)
  end

  # Removes /s from resource names; this allows Go package names
  # to be used as resource names without confusing software that
  # interacts with {download_name}, e.g. `github.com/foo/bar`.
  def escaped_name
    name.tr("/", "-")
  end

  def download_name
    return owner.name if name.nil?
    return escaped_name if owner.nil?

    "#{owner.name}--#{escaped_name}"
  end

  def downloaded?
    cached_download.exist?
  end

  def cached_download
    downloader.cached_location
  end

  def clear_cache
    downloader.clear_cache
  end

  # Verifies download and unpacks it.
  # The block may call `|resource, staging| staging.retain!` to retain the staging
  # directory. Subclasses that override stage should implement the tmp
  # dir using {Mktemp} so that works with all subtypes.
  #
  # @api public
  def stage(target = nil, &block)
    raise ArgumentError, "target directory or block is required" if !target && block.blank?

    prepare_patches
    fetch_patches(skip_downloaded: true)
    fetch unless downloaded?

    unpack(target, &block)
  end

  def prepare_patches
    patches.grep(DATAPatch) { |p| p.path = owner.owner.path }
  end

  def fetch_patches(skip_downloaded: false)
    external_patches = patches.select(&:external?)
    external_patches.reject!(&:downloaded?) if skip_downloaded
    external_patches.each(&:fetch)
  end

  def apply_patches
    return if patches.empty?

    ohai "Patching #{name}"
    patches.each(&:apply)
  end

  # If a target is given, unpack there; else unpack to a temp folder.
  # If block is given, yield to that block with `|stage|`, where stage
  # is a {ResourceStageContext}.
  # A target or a block must be given, but not both.
  def unpack(target = nil)
    mktemp(download_name) do |staging|
      downloader.stage do
        @source_modified_time = downloader.source_modified_time
        apply_patches
        if block_given?
          yield ResourceStageContext.new(self, staging)
        elsif target
          target = Pathname(target)
          target.install Pathname.pwd.children
        end
      end
    end
  end

  Partial = Struct.new(:resource, :files)

  def files(*files)
    Partial.new(self, files)
  end

  def fetch(verify_download_integrity: true)
    HOMEBREW_CACHE.mkpath

    fetch_patches

    begin
      downloader.fetch
    rescue ErrorDuringExecution, CurlDownloadStrategyError => e
      raise DownloadError.new(self, e)
    end

    download = cached_download
    verify_download_integrity(download) if verify_download_integrity
    download
  end

  def verify_download_integrity(fn)
    if fn.file?
      ohai "Verifying checksum for '#{fn.basename}'" if verbose?
      fn.verify_checksum(checksum)
    end
  rescue ChecksumMissingError
    opoo <<~EOS
      Cannot verify integrity of '#{fn.basename}'.
      No checksum was provided for this resource.
      For your reference, the checksum is:
        sha256 "#{fn.sha256}"
    EOS
  end

  def sha256(val)
    @checksum = Checksum.new(val)
  end

  def url(val = nil, **specs)
    return @url if val.nil?

    @url = val
    @specs.merge!(specs)
    @using = @specs.delete(:using)
    @download_strategy = DownloadStrategyDetector.detect(url, using)
    @downloader = nil
  end

  def version(val = nil)
    @version ||= begin
      version = detect_version(val)
      version.null? ? nil : version
    end
  end

  def mirror(val)
    mirrors << val
  end

  def patch(strip = :p1, src = nil, &block)
    p = Patch.create(strip, src, &block)
    patches << p
  end

  protected

  def mktemp(prefix, &block)
    Mktemp.new(prefix).run(&block)
  end

  private

  def detect_version(val)
    return Version::NULL if val.nil? && url.nil?

    case val
    when nil     then Version.detect(url, **specs)
    when String  then Version.create(val)
    when Version then val
    else
      raise TypeError, "version '#{val.inspect}' should be a string"
    end
  end

  # A resource containing a Go package.
  class Go < Resource
    def stage(target, &block)
      super(target/name, &block)
    end
  end

  # A resource containing a patch.
  class PatchResource < Resource
    attr_reader :patch_files

    def initialize(&block)
      @patch_files = []
      @directory = nil
      super "patch", &block
    end

    def apply(*paths)
      paths.flatten!
      @patch_files.concat(paths)
      @patch_files.uniq!
    end

    def directory(val = nil)
      return @directory if val.nil?

      @directory = val
    end
  end
end

# The context in which a {Resource#stage} occurs. Supports access to both
# the {Resource} and associated {Mktemp} in a single block argument. The interface
# is back-compatible with {Resource} itself as used in that context.
#
# @api private
class ResourceStageContext
  extend T::Sig

  extend Forwardable

  # The {Resource} that is being staged.
  attr_reader :resource
  # The {Mktemp} in which {#resource} is staged.
  attr_reader :staging

  def_delegators :@resource, :version, :url, :mirrors, :specs, :using, :source_modified_time
  def_delegators :@staging, :retain!

  def initialize(resource, staging)
    @resource = resource
    @staging = staging
  end

  sig { returns(String) }
  def to_s
    "<#{self.class}: resource=#{resource} staging=#{staging}>"
  end
end
