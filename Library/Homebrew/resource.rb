# typed: true
# frozen_string_literal: true

require "download_strategy"
require "checksum"
require "version"
require "mktemp"
require "livecheck"
require "extend/on_system"

# Resource is the fundamental representation of an external resource. The
# primary formula download, along with other declared resources, are instances
# of this class.
#
# @api private
class Resource
  extend T::Sig

  include Context
  include FileUtils
  include OnSystem::MacOSAndLinux

  attr_reader :mirrors, :specs, :using, :source_modified_time, :patches, :owner
  attr_writer :version
  attr_accessor :download_strategy, :checksum

  # Formula name must be set after the DSL, as we have no access to the
  # formula name before initialization of the formula.
  attr_accessor :name

  def initialize(name = nil, &block)
    # Ensure this is synced with `initialize_dup` and `freeze` (excluding simple objects like integers and booleans)
    @name = name
    @url = nil
    @version = nil
    @mirrors = []
    @specs = {}
    @checksum = nil
    @using = nil
    @patches = []
    @livecheck = Livecheck.new(self)
    @livecheckable = false
    instance_eval(&block) if block
  end

  def initialize_dup(other)
    super
    @name = @name.dup
    @version = @version.dup
    @mirrors = @mirrors.dup
    @specs = @specs.dup
    @checksum = @checksum.dup
    @using = @using.dup
    @patches = @patches.dup
    @livecheck = @livecheck.dup
  end

  def freeze
    @name.freeze
    @version.freeze
    @mirrors.freeze
    @specs.freeze
    @checksum.freeze
    @using.freeze
    @patches.freeze
    @livecheck.freeze
    super
  end

  def owner=(owner)
    @owner = owner
    patches.each { |p| p.owner = owner }

    return if !owner.respond_to?(:full_name) || owner.full_name != "ca-certificates"
    return if Homebrew::EnvConfig.no_insecure_redirect?

    @specs[:insecure] = !specs[:bottle] && !DevelopmentTools.ca_file_handles_most_https_certificates?
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
  def stage(target = nil, debug_symbols: false, &block)
    raise ArgumentError, "target directory or block is required" if !target && block.blank?

    prepare_patches
    fetch_patches(skip_downloaded: true)
    fetch unless downloaded?

    unpack(target, debug_symbols: debug_symbols, &block)
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
  def unpack(target = nil, debug_symbols: false)
    current_working_directory = Pathname.pwd
    stage_resource(download_name, debug_symbols: debug_symbols) do |staging|
      downloader.stage do
        @source_modified_time = downloader.source_modified_time
        apply_patches
        if block_given?
          yield ResourceStageContext.new(self, staging)
        elsif target
          target = Pathname(target)
          target = current_working_directory/target if target.relative?
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

  # @!attribute [w] livecheck
  # {Livecheck} can be used to check for newer versions of the software.
  # This method evaluates the DSL specified in the livecheck block of the
  # {Resource} (if it exists) and sets the instance variables of a {Livecheck}
  # object accordingly. This is used by `brew livecheck` to check for newer
  # versions of the software.
  #
  # <pre>livecheck do
  #   url "https://example.com/foo/releases"
  #   regex /foo-(\d+(?:\.\d+)+)\.tar/
  # end</pre>
  def livecheck(&block)
    return @livecheck unless block

    @livecheckable = true
    @livecheck.instance_eval(&block)
  end

  # Whether a livecheck specification is defined or not.
  # It returns true when a livecheck block is present in the {Resource} and
  # false otherwise, and is used by livecheck.
  def livecheckable?
    @livecheckable == true
  end

  def sha256(val)
    @checksum = Checksum.new(val)
  end

  def url(val = nil, **specs)
    return @url if val.nil?

    specs = specs.dup
    # Don't allow this to be set.
    specs.delete(:insecure)

    @url = val
    @using = specs.delete(:using)
    @download_strategy = DownloadStrategyDetector.detect(url, using)
    @specs.merge!(specs)
    @downloader = nil
    @version = detect_version(@version)
  end

  def version(val = nil)
    return @version if val.nil?

    @version = detect_version(val)
  end

  def mirror(val)
    mirrors << val
  end

  def patch(strip = :p1, src = nil, &block)
    p = Patch.create(strip, src, &block)
    patches << p
  end

  protected

  def stage_resource(prefix, debug_symbols: false, &block)
    Mktemp.new(prefix, retain_in_cache: debug_symbols).run(&block)
  end

  private

  def detect_version(val)
    version = case val
    when nil     then url.nil? ? Version::NULL : Version.detect(url, **specs)
    when String  then Version.create(val)
    when Version then val
    else
      raise TypeError, "version '#{val.inspect}' should be a string"
    end

    version unless version.null?
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
