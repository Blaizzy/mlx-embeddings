# typed: false
# frozen_string_literal: true

require "json"
require "time"
require "unpack_strategy"
require "lazy_object"
require "cgi"
require "lock_file"

require "mechanize/version"
require "mechanize/http/content_disposition_parser"

require "utils/curl"

# @abstract Abstract superclass for all download strategies.
#
# @api private
class AbstractDownloadStrategy
  extend T::Sig

  extend Forwardable
  include FileUtils
  include Context

  # Extension for bottle downloads.
  #
  # @api private
  module Pourable
    def stage
      ohai "Pouring #{basename}"
      super
    end
  end

  attr_reader :cache, :cached_location, :url, :meta, :name, :version

  private :meta, :name, :version

  def initialize(url, name, version, **meta)
    @url = url
    @name = name
    @version = version
    @cache = meta.fetch(:cache, HOMEBREW_CACHE)
    @meta = meta
    @quiet = false
    extend Pourable if meta[:bottle]
  end

  # Download and cache the resource at {#cached_location}.
  #
  # @api public
  def fetch; end

  # Disable any output during downloading.
  #
  # TODO: Deprecate once we have an explicitly documented alternative.
  #
  # @api public
  sig { void }
  def shutup!
    @quiet = true
  end

  def quiet?
    Context.current.quiet? || @quiet
  end

  # Unpack {#cached_location} into the current working directory.
  #
  # Additionally, if a block is given, the working directory was previously empty
  # and a single directory is extracted from the archive, the block will be called
  # with the working directory changed to that directory. Otherwise this method
  # will return, or the block will be called, without changing the current working
  # directory.
  #
  # @api public
  def stage(&block)
    UnpackStrategy.detect(cached_location,
                          prioritise_extension: true,
                          ref_type: @ref_type, ref: @ref)
                  .extract_nestedly(basename:             basename,
                                    prioritise_extension: true,
                                    verbose:              verbose? && !quiet?)
    chdir(&block) if block
  end

  def chdir(&block)
    entries = Dir["*"]
    raise "Empty archive" if entries.length.zero?

    if entries.length != 1
      yield
      return
    end

    if File.directory? entries.first
      Dir.chdir(entries.first, &block)
    else
      yield
    end
  end
  private :chdir

  # @!attribute [r] source_modified_time
  # Returns the most recent modified time for all files in the current working directory after stage.
  #
  # @api public
  def source_modified_time
    Pathname.pwd.to_enum(:find).select(&:file?).map(&:mtime).max
  end

  # Remove {#cached_location} and any other files associated with the resource
  # from the cache.
  #
  # @api public
  def clear_cache
    rm_rf(cached_location)
  end

  def basename
    cached_location.basename
  end

  private

  def puts(*args)
    super(*args) unless quiet?
  end

  def ohai(*args)
    super(*args) unless quiet?
  end

  def silent_command(*args, **options)
    system_command(*args, print_stderr: false, env: env, **options)
  end

  def command!(*args, **options)
    system_command!(
      *args,
      env: env.merge(options.fetch(:env, {})),
      **command_output_options,
      **options,
    )
  end

  def command_output_options
    {
      print_stdout: !quiet?,
      print_stderr: !quiet?,
      verbose:      verbose? && !quiet?,
    }
  end

  def env
    {}
  end
end

# @abstract Abstract superclass for all download strategies downloading from a version control system.
#
# @api private
class VCSDownloadStrategy < AbstractDownloadStrategy
  REF_TYPES = [:tag, :branch, :revisions, :revision].freeze

  def initialize(url, name, version, **meta)
    super
    @ref_type, @ref = extract_ref(meta)
    @revision = meta[:revision]
    @cached_location = @cache/"#{name}--#{cache_tag}"
  end

  # Download and cache the repository at {#cached_location}.
  #
  # @api public
  def fetch
    ohai "Cloning #{url}"

    if cached_location.exist? && repo_valid?
      puts "Updating #{cached_location}"
      update
    elsif cached_location.exist?
      puts "Removing invalid repository from cache"
      clear_cache
      clone_repo
    else
      clone_repo
    end

    version.update_commit(last_commit) if head?

    return if @ref_type != :tag || @revision.blank? || current_revision.blank? || current_revision == @revision

    raise <<~EOS
      #{@ref} tag should be #{@revision}
      but is actually #{current_revision}
    EOS
  end

  def fetch_last_commit
    fetch
    last_commit
  end

  def commit_outdated?(commit)
    @last_commit ||= fetch_last_commit
    commit != @last_commit
  end

  def head?
    version.respond_to?(:head?) && version.head?
  end

  # Return last commit's unique identifier for the repository.
  # Return most recent modified timestamp unless overridden.
  #
  # @api public
  def last_commit
    source_modified_time.to_i.to_s
  end

  private

  def cache_tag
    raise NotImplementedError
  end

  def repo_valid?
    raise NotImplementedError
  end

  def clone_repo; end

  def update; end

  def current_revision; end

  def extract_ref(specs)
    key = REF_TYPES.find { |type| specs.key?(type) }
    [key, specs[key]]
  end
end

# @abstract Abstract superclass for all download strategies downloading a single file.
#
# @api private
class AbstractFileDownloadStrategy < AbstractDownloadStrategy
  # Path for storing an incomplete download while the download is still in progress.
  #
  # @api public
  def temporary_path
    @temporary_path ||= Pathname.new("#{cached_location}.incomplete")
  end

  # Path of the symlink (whose name includes the resource name, version and extension)
  # pointing to {#cached_location}.
  #
  # @api public
  def symlink_location
    return @symlink_location if defined?(@symlink_location)

    ext = Pathname(parse_basename(url)).extname
    @symlink_location = @cache/"#{name}--#{version}#{ext}"
  end

  # Path for storing the completed download.
  #
  # @api public
  def cached_location
    return @cached_location if defined?(@cached_location)

    url_sha256 = Digest::SHA256.hexdigest(url)
    downloads = Pathname.glob(HOMEBREW_CACHE/"downloads/#{url_sha256}--*")
                        .reject { |path| path.extname.end_with?(".incomplete") }

    @cached_location = if downloads.count == 1
      downloads.first
    else
      HOMEBREW_CACHE/"downloads/#{url_sha256}--#{resolved_basename}"
    end
  end

  def basename
    cached_location.basename.sub(/^[\da-f]{64}--/, "")
  end

  private

  def resolved_url
    resolved_url, = resolved_url_and_basename
    resolved_url
  end

  def resolved_basename
    _, resolved_basename = resolved_url_and_basename
    resolved_basename
  end

  def resolved_url_and_basename
    return @resolved_url_and_basename if defined?(@resolved_url_and_basename)

    @resolved_url_and_basename = [url, parse_basename(url)]
  end

  def parse_basename(url)
    uri_path = if url.match?(URI::DEFAULT_PARSER.make_regexp)
      uri = URI(url)

      if uri.query
        query_params = CGI.parse(uri.query)
        query_params["response-content-disposition"].each do |param|
          query_basename = param[/attachment;\s*filename=(["']?)(.+)\1/i, 2]
          return query_basename if query_basename
        end
      end

      uri.query ? "#{uri.path}?#{uri.query}" : uri.path
    else
      url
    end

    uri_path = URI.decode_www_form_component(uri_path)

    # We need a Pathname because we've monkeypatched extname to support double
    # extensions (e.g. tar.gz).
    # Given a URL like https://example.com/download.php?file=foo-1.0.tar.gz
    # the basename we want is "foo-1.0.tar.gz", not "download.php".
    Pathname.new(uri_path).ascend do |path|
      ext = path.extname[/[^?&]+/]
      return path.basename.to_s[/[^?&]+#{Regexp.escape(ext)}/] if ext
    end

    File.basename(uri_path)
  end
end

# Strategy for downloading files using `curl`.
#
# @api public
class CurlDownloadStrategy < AbstractFileDownloadStrategy
  include Utils::Curl

  attr_reader :mirrors

  def initialize(url, name, version, **meta)
    super
    @mirrors = meta.fetch(:mirrors, [])
  end

  # Download and cache the file at {#cached_location}.
  #
  # @api public
  def fetch
    download_lock = LockFile.new(temporary_path.basename)
    download_lock.lock

    urls = [url, *mirrors]

    begin
      url = urls.shift

      ohai "Downloading #{url}"

      resolved_url, _, url_time, = resolve_url_basename_time_file_size(url)

      fresh = if cached_location.exist? && url_time
        url_time <= cached_location.mtime
      elsif version.respond_to?(:latest?)
        !version.latest?
      else
        true
      end

      if cached_location.exist? && fresh
        puts "Already downloaded: #{cached_location}"
      else
        begin
          _fetch(url: url, resolved_url: resolved_url)
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
    rescue CurlDownloadStrategyError
      raise if urls.empty?

      puts "Trying a mirror..."
      retry
    end
  ensure
    download_lock&.unlock
    download_lock&.path&.unlink
  end

  def clear_cache
    super
    rm_rf(temporary_path)
  end

  def resolved_time_file_size
    _, _, time, file_size = resolve_url_basename_time_file_size(url)
    [time, file_size]
  end

  private

  def resolved_url_and_basename
    resolved_url, basename, = resolve_url_basename_time_file_size(url)
    [resolved_url, basename]
  end

  def resolve_url_basename_time_file_size(url)
    @resolved_info_cache ||= {}
    return @resolved_info_cache[url] if @resolved_info_cache.include?(url)

    if (domain = Homebrew::EnvConfig.artifact_domain)
      url = url.sub(%r{^((ht|f)tps?://)?}, "#{domain.chomp("/")}/")
    end

    out, _, status= curl_output("--location", "--silent", "--head", "--request", "GET", url.to_s)

    lines = status.success? ? out.lines.map(&:chomp) : []

    locations = lines.map { |line| line[/^Location:\s*(.*)$/i, 1] }
                     .compact

    redirect_url = locations.reduce(url) do |current_url, location|
      if location.start_with?("//")
        uri = URI(current_url)
        "#{uri.scheme}:#{location}"
      elsif location.start_with?("/")
        uri = URI(current_url)
        "#{uri.scheme}://#{uri.host}#{location}"
      elsif location.start_with?("./")
        uri = URI(current_url)
        "#{uri.scheme}://#{uri.host}#{Pathname(uri.path).dirname/location}"
      else
        location
      end
    end

    content_disposition_parser = Mechanize::HTTP::ContentDispositionParser.new

    parse_content_disposition = lambda do |line|
      next unless (content_disposition = content_disposition_parser.parse(line.sub(/; *$/, ""), true))

      filename = nil

      if (filename_with_encoding = content_disposition.parameters["filename*"])
        encoding, encoded_filename = filename_with_encoding.split("''", 2)
        filename = URI.decode_www_form_component(encoded_filename).encode(encoding) if encoding && encoded_filename
      end

      filename || content_disposition.filename
    end

    filenames = lines.map(&parse_content_disposition).compact

    time =
      lines.map { |line| line[/^Last-Modified:\s*(.+)/i, 1] }
           .compact
           .map { |t| t.match?(/^\d+$/) ? Time.at(t.to_i) : Time.parse(t) }
           .last

    file_size =
      lines.map { |line| line[/^Content-Length:\s*(\d+)/i, 1] }
           .compact
           .map(&:to_i)
           .last

    basename = filenames.last || parse_basename(redirect_url)

    @resolved_info_cache[url] = [redirect_url, basename, time, file_size]
  end

  def _fetch(url:, resolved_url:)
    ohai "Downloading from #{resolved_url}" if url != resolved_url

    if Homebrew::EnvConfig.no_insecure_redirect? &&
       url.start_with?("https://") && !resolved_url.start_with?("https://")
      $stderr.puts "HTTPS to HTTP redirect detected & HOMEBREW_NO_INSECURE_REDIRECT is set."
      raise CurlDownloadStrategyError, url
    end

    curl_download resolved_url, to: temporary_path
  end

  # Curl options to be always passed to curl,
  # with raw head calls (`curl --head`) or with actual `fetch`.
  def _curl_args
    args = []

    args += ["-b", meta.fetch(:cookies).map { |k, v| "#{k}=#{v}" }.join(";")] if meta.key?(:cookies)

    args += ["-e", meta.fetch(:referer)] if meta.key?(:referer)

    args += ["--user", meta.fetch(:user)] if meta.key?(:user)

    args += [meta[:header], meta[:headers]].flatten.compact.flat_map { |h| ["--header", h.strip] }

    args
  end

  def _curl_opts
    return { user_agent: meta.fetch(:user_agent) } if meta.key?(:user_agent)

    {}
  end

  def curl_output(*args, **options)
    super(*_curl_args, *args, **_curl_opts, **options)
  end

  def curl(*args, **options)
    args << "--connect-timeout" << "15" unless mirrors.empty?
    super(*_curl_args, *args, **_curl_opts, **command_output_options, **options)
  end
end

# Strategy for downloading a file from an Apache Mirror URL.
#
# @api public
class CurlApacheMirrorDownloadStrategy < CurlDownloadStrategy
  def mirrors
    combined_mirrors
  end

  private

  def combined_mirrors
    return @combined_mirrors if defined?(@combined_mirrors)

    backup_mirrors = apache_mirrors.fetch("backup", [])
                                   .map { |mirror| "#{mirror}#{apache_mirrors["path_info"]}" }

    @combined_mirrors = [*@mirrors, *backup_mirrors]
  end

  def resolve_url_basename_time_file_size(url)
    if url == self.url
      super("#{apache_mirrors["preferred"]}#{apache_mirrors["path_info"]}")
    else
      super
    end
  end

  def apache_mirrors
    return @apache_mirrors if defined?(@apache_mirrors)

    json, = curl_output("--silent", "--location", "#{url}&asjson=1")
    @apache_mirrors = JSON.parse(json)
  rescue JSON::ParserError
    raise CurlDownloadStrategyError, "Couldn't determine mirror, try again later."
  end
end

# Strategy for downloading via an HTTP POST request using `curl`.
# Query parameters on the URL are converted into POST parameters.
#
# @api public
class CurlPostDownloadStrategy < CurlDownloadStrategy
  private

  def _fetch(url:, resolved_url:)
    args = if meta.key?(:data)
      escape_data = ->(d) { ["-d", URI.encode_www_form([d])] }
      [url, *meta[:data].flat_map(&escape_data)]
    else
      url, query = url.split("?", 2)
      query.nil? ? [url, "-X", "POST"] : [url, "-d", query]
    end

    curl_download(*args, to: temporary_path)
  end
end

# Strategy for downloading archives without automatically extracting them.
# (Useful for downloading `.jar` files.)
#
# @api public
class NoUnzipCurlDownloadStrategy < CurlDownloadStrategy
  def stage
    UnpackStrategy::Uncompressed.new(cached_location)
                                .extract(basename: basename,
                                         verbose:  verbose? && !quiet?)
    yield if block_given?
  end
end

# Strategy for extracting local binary packages.
#
# @api private
class LocalBottleDownloadStrategy < AbstractFileDownloadStrategy
  def initialize(path) # rubocop:disable Lint/MissingSuper
    @cached_location = path
  end
end

# Strategy for downloading a Subversion repository.
#
# @api public
class SubversionDownloadStrategy < VCSDownloadStrategy
  extend T::Sig

  def initialize(url, name, version, **meta)
    super
    @url = @url.sub("svn+http://", "")
  end

  # Download and cache the repository at {#cached_location}.
  #
  # @api public
  def fetch
    if @url.chomp("/") != repo_url || !silent_command("svn", args: ["switch", @url, cached_location]).success?
      clear_cache
    end
    super
  end

  # (see AbstractDownloadStrategy#source_modified_time)
  sig { returns(Time) }
  def source_modified_time
    time = if Version.create(Utils::Svn.version) >= Version.create("1.9")
      out, = silent_command("svn", args: ["info", "--show-item", "last-changed-date"], chdir: cached_location)
      out
    else
      out, = silent_command("svn", args: ["info"], chdir: cached_location)
      out[/^Last Changed Date: (.+)$/, 1]
    end
    Time.parse time
  end

  # (see VCSDownloadStrategy#source_modified_time)
  def last_commit
    out, = silent_command("svn", args: ["info", "--show-item", "revision"], chdir: cached_location)
    out.strip
  end

  private

  def repo_url
    out, = silent_command("svn", args: ["info"], chdir: cached_location)
    out.strip[/^URL: (.+)$/, 1]
  end

  def externals
    out, = silent_command("svn", args: ["propget", "svn:externals", @url])
    out.chomp.split("\n").each do |line|
      name, url = line.split(/\s+/)
      yield name, url
    end
  end

  def fetch_repo(target, url, revision = nil, ignore_externals: false)
    # Use "svn update" when the repository already exists locally.
    # This saves on bandwidth and will have a similar effect to verifying the
    # cache as it will make any changes to get the right revision.
    args = []
    args << "--quiet" unless verbose?

    if revision
      ohai "Checking out #{@ref}"
      args << "-r" << revision
    end

    args << "--ignore-externals" if ignore_externals

    if meta[:trust_cert] == true
      args << "--trust-server-cert"
      args << "--non-interactive"
    end

    if target.directory?
      command!("svn", args: ["update", *args], chdir: target.to_s)
    else
      command!("svn", args: ["checkout", url, target, *args])
    end
  end

  sig { returns(String) }
  def cache_tag
    head? ? "svn-HEAD" : "svn"
  end

  def repo_valid?
    (cached_location/".svn").directory?
  end

  def clone_repo
    case @ref_type
    when :revision
      fetch_repo cached_location, @url, @ref
    when :revisions
      # nil is OK for main_revision, as fetch_repo will then get latest
      main_revision = @ref[:trunk]
      fetch_repo cached_location, @url, main_revision, ignore_externals: true

      externals do |external_name, external_url|
        fetch_repo cached_location/external_name, external_url, @ref[external_name], ignore_externals: true
      end
    else
      fetch_repo cached_location, @url
    end
  end
  alias update clone_repo
end

# Strategy for downloading a Git repository.
#
# @api public
class GitDownloadStrategy < VCSDownloadStrategy
  SHALLOW_CLONE_ALLOWLIST = [
    %r{git://},
    %r{https://github\.com},
    %r{http://git\.sv\.gnu\.org},
    %r{http://llvm\.org},
  ].freeze

  def initialize(url, name, version, **meta)
    super
    @ref_type ||= :branch
    @ref ||= "master"
    @shallow = meta.fetch(:shallow, true)
  end

  # (see AbstractDownloadStrategy#source_modified_time)
  sig { returns(Time) }
  def source_modified_time
    out, = silent_command("git", args: ["--git-dir", git_dir, "show", "-s", "--format=%cD"])
    Time.parse(out)
  end

  # (see VCSDownloadStrategy#source_modified_time)
  def last_commit
    out, = silent_command("git", args: ["--git-dir", git_dir, "rev-parse", "--short=7", "HEAD"])
    out.chomp
  end

  private

  sig { returns(String) }
  def cache_tag
    "git"
  end

  sig { returns(Integer) }
  def cache_version
    0
  end

  def update
    config_repo
    update_repo
    checkout
    reset
    update_submodules if submodules?
  end

  def shallow_clone?
    @shallow && support_depth?
  end

  def shallow_dir?
    (git_dir/"shallow").exist?
  end

  def support_depth?
    @ref_type != :revision && SHALLOW_CLONE_ALLOWLIST.any? { |regex| @url =~ regex }
  end

  def git_dir
    cached_location/".git"
  end

  def ref?
    silent_command("git",
                   args: ["--git-dir", git_dir, "rev-parse", "-q", "--verify", "#{@ref}^{commit}"])
      .success?
  end

  def current_revision
    out, = silent_command("git", args: ["--git-dir", git_dir, "rev-parse", "-q", "--verify", "HEAD"])
    out.strip
  end

  def repo_valid?
    silent_command("git", args: ["--git-dir", git_dir, "status", "-s"]).success?
  end

  def submodules?
    (cached_location/".gitmodules").exist?
  end

  sig { returns(T::Array[String]) }
  def clone_args
    args = %w[clone]
    args << "--depth" << "1" if shallow_clone?

    case @ref_type
    when :branch, :tag
      args << "--branch" << @ref
      args << "-c" << "advice.detachedHead=false" # silences detached head warning
    end

    args << @url << cached_location
  end

  sig { returns(String) }
  def refspec
    case @ref_type
    when :branch then "+refs/heads/#{@ref}:refs/remotes/origin/#{@ref}"
    when :tag    then "+refs/tags/#{@ref}:refs/tags/#{@ref}"
    else              "+refs/heads/master:refs/remotes/origin/master"
    end
  end

  def config_repo
    command! "git",
             args:  ["config", "remote.origin.url", @url],
             chdir: cached_location
    command! "git",
             args:  ["config", "remote.origin.fetch", refspec],
             chdir: cached_location
    command! "git",
             args:  ["config", "remote.origin.tagOpt", "--no-tags"],
             chdir: cached_location
  end

  def update_repo
    return if @ref_type != :branch && ref?

    if !shallow_clone? && shallow_dir?
      command! "git",
               args:  ["fetch", "origin", "--unshallow"],
               chdir: cached_location
    else
      command! "git",
               args:  ["fetch", "origin"],
               chdir: cached_location
    end
  end

  def clone_repo
    command! "git", args: clone_args

    command! "git",
             args:  ["config", "homebrew.cacheversion", cache_version],
             chdir: cached_location
    checkout
    update_submodules if submodules?
  end

  def checkout
    ohai "Checking out #{@ref_type} #{@ref}" if @ref_type && @ref
    command! "git", args: ["checkout", "-f", @ref, "--"], chdir: cached_location
  end

  def reset
    ref = case @ref_type
    when :branch
      "origin/#{@ref}"
    when :revision, :tag
      @ref
    end

    command! "git",
             args:  ["reset", "--hard", *ref, "--"],
             chdir: cached_location
  end

  def update_submodules
    command! "git",
             args:  ["submodule", "foreach", "--recursive", "git submodule sync"],
             chdir: cached_location
    command! "git",
             args:  ["submodule", "update", "--init", "--recursive"],
             chdir: cached_location
    fix_absolute_submodule_gitdir_references!
  end

  # When checking out Git repositories with recursive submodules, some Git
  # versions create `.git` files with absolute instead of relative `gitdir:`
  # pointers. This works for the cached location, but breaks various Git
  # operations once the affected Git resource is staged, i.e. recursively
  # copied to a new location. (This bug was introduced in Git 2.7.0 and fixed
  # in 2.8.3. Clones created with affected version remain broken.)
  # See https://github.com/Homebrew/homebrew-core/pull/1520 for an example.
  def fix_absolute_submodule_gitdir_references!
    submodule_dirs = command!("git",
                              args:  ["submodule", "--quiet", "foreach", "--recursive", "pwd"],
                              chdir: cached_location).stdout

    submodule_dirs.lines.map(&:chomp).each do |submodule_dir|
      work_dir = Pathname.new(submodule_dir)

      # Only check and fix if `.git` is a regular file, not a directory.
      dot_git = work_dir/".git"
      next unless dot_git.file?

      git_dir = dot_git.read.chomp[/^gitdir: (.*)$/, 1]
      if git_dir.nil?
        onoe "Failed to parse '#{dot_git}'." if Homebrew::EnvConfig.developer?
        next
      end

      # Only attempt to fix absolute paths.
      next unless git_dir.start_with?("/")

      # Make the `gitdir:` reference relative to the working directory.
      relative_git_dir = Pathname.new(git_dir).relative_path_from(work_dir)
      dot_git.atomic_write("gitdir: #{relative_git_dir}\n")
    end
  end
end

# Strategy for downloading a Git repository from GitHub.
#
# @api public
class GitHubGitDownloadStrategy < GitDownloadStrategy
  def initialize(url, name, version, **meta)
    super

    return unless %r{^https?://github\.com/(?<user>[^/]+)/(?<repo>[^/]+)\.git$} =~ @url

    @user = user
    @repo = repo
  end

  def github_last_commit
    return if Homebrew::EnvConfig.no_github_api?

    output, _, status = curl_output(
      "--silent", "--head", "--location",
      "-H", "Accept: application/vnd.github.v3.sha",
      "https://api.github.com/repos/#{@user}/#{@repo}/commits/#{@ref}"
    )

    return unless status.success?

    commit = output[/^ETag: "(\h+)"/, 1]
    version.update_commit(commit) if commit
    commit
  end

  def multiple_short_commits_exist?(commit)
    return if Homebrew::EnvConfig.no_github_api?

    output, _, status = curl_output(
      "--silent", "--head", "--location",
      "-H", "Accept: application/vnd.github.v3.sha",
      "https://api.github.com/repos/#{@user}/#{@repo}/commits/#{commit}"
    )

    !(status.success? && output && output[/^Status: (200)/, 1] == "200")
  end

  def commit_outdated?(commit)
    @last_commit ||= github_last_commit
    if @last_commit
      return true unless commit
      return true unless @last_commit.start_with?(commit)

      if multiple_short_commits_exist?(commit)
        true
      else
        version.update_commit(commit)
        false
      end
    else
      super
    end
  end
end

# Strategy for downloading a CVS repository.
#
# @api public
class CVSDownloadStrategy < VCSDownloadStrategy
  extend T::Sig

  def initialize(url, name, version, **meta)
    super
    @url = @url.sub(%r{^cvs://}, "")

    if meta.key?(:module)
      @module = meta.fetch(:module)
    elsif !@url.match?(%r{:[^/]+$})
      @module = name
    else
      @module, @url = split_url(@url)
    end
  end

  # (see AbstractDownloadStrategy#source_modified_time)
  sig { returns(Time) }
  def source_modified_time
    # Filter CVS's files because the timestamp for each of them is the moment
    # of clone.
    max_mtime = Time.at(0)
    cached_location.find do |f|
      Find.prune if f.directory? && f.basename.to_s == "CVS"
      next unless f.file?

      mtime = f.mtime
      max_mtime = mtime if mtime > max_mtime
    end
    max_mtime
  end

  private

  def env
    { "PATH" => PATH.new("/usr/bin", Formula["cvs"].opt_bin, ENV["PATH"]) }
  end

  sig { returns(String) }
  def cache_tag
    "cvs"
  end

  def repo_valid?
    (cached_location/"CVS").directory?
  end

  def quiet_flag
    "-Q" unless verbose?
  end

  def clone_repo
    # Login is only needed (and allowed) with pserver; skip for anoncvs.
    command! "cvs", args: [*quiet_flag, "-d", @url, "login"] if @url.include? "pserver"

    command! "cvs",
             args:  [*quiet_flag, "-d", @url, "checkout", "-d", cached_location.basename, @module],
             chdir: cached_location.dirname
  end

  def update
    command! "cvs",
             args:  [*quiet_flag, "update"],
             chdir: cached_location
  end

  def split_url(in_url)
    parts = in_url.split(/:/)
    mod = parts.pop
    url = parts.join(":")
    [mod, url]
  end
end

# Strategy for downloading a Mercurial repository.
#
# @api public
class MercurialDownloadStrategy < VCSDownloadStrategy
  extend T::Sig

  def initialize(url, name, version, **meta)
    super
    @url = @url.sub(%r{^hg://}, "")
  end

  # (see AbstractDownloadStrategy#source_modified_time)
  sig { returns(Time) }
  def source_modified_time
    out, = silent_command("hg",
                          args: ["tip", "--template", "{date|isodate}", "-R", cached_location])

    Time.parse(out)
  end

  # (see VCSDownloadStrategy#source_modified_time)
  def last_commit
    out, = silent_command("hg", args: ["parent", "--template", "{node|short}", "-R", cached_location])
    out.chomp
  end

  private

  def env
    { "PATH" => PATH.new(Formula["mercurial"].opt_bin, ENV["PATH"]) }
  end

  sig { returns(String) }
  def cache_tag
    "hg"
  end

  def repo_valid?
    (cached_location/".hg").directory?
  end

  def clone_repo
    command! "hg", args: ["clone", @url, cached_location]
  end

  def update
    command! "hg", args: ["--cwd", cached_location, "pull", "--update"]

    update_args = if @ref_type && @ref
      ohai "Checking out #{@ref_type} #{@ref}"
      [@ref]
    else
      ["--clean"]
    end

    command! "hg", args: ["--cwd", cached_location, "update", *update_args]
  end
end

# Strategy for downloading a Bazaar repository.
#
# @api public
class BazaarDownloadStrategy < VCSDownloadStrategy
  extend T::Sig

  def initialize(url, name, version, **meta)
    super
    @url.sub!(%r{^bzr://}, "")
  end

  # (see AbstractDownloadStrategy#source_modified_time)
  sig { returns(Time) }
  def source_modified_time
    out, = silent_command("bzr", args: ["log", "-l", "1", "--timezone=utc", cached_location])
    timestamp = out.chomp
    raise "Could not get any timestamps from bzr!" if timestamp.blank?

    Time.parse(timestamp)
  end

  # (see VCSDownloadStrategy#source_modified_time)
  def last_commit
    out, = silent_command("bzr", args: ["revno", cached_location])
    out.chomp
  end

  private

  def env
    {
      "PATH"     => PATH.new(Formula["bazaar"].opt_bin, ENV["PATH"]),
      "BZR_HOME" => HOMEBREW_TEMP,
    }
  end

  sig { returns(String) }
  def cache_tag
    "bzr"
  end

  def repo_valid?
    (cached_location/".bzr").directory?
  end

  def clone_repo
    # "lightweight" means history-less
    command! "bzr",
             args: ["checkout", "--lightweight", @url, cached_location]
  end

  def update
    command! "bzr",
             args:  ["update"],
             chdir: cached_location
  end
end

# Strategy for downloading a Fossil repository.
#
# @api public
class FossilDownloadStrategy < VCSDownloadStrategy
  extend T::Sig

  def initialize(url, name, version, **meta)
    super
    @url = @url.sub(%r{^fossil://}, "")
  end

  # (see AbstractDownloadStrategy#source_modified_time)
  sig { returns(Time) }
  def source_modified_time
    out, = silent_command("fossil", args: ["info", "tip", "-R", cached_location])
    Time.parse(out[/^uuid: +\h+ (.+)$/, 1])
  end

  # (see VCSDownloadStrategy#source_modified_time)
  def last_commit
    out, = silent_command("fossil", args: ["info", "tip", "-R", cached_location])
    out[/^uuid: +(\h+) .+$/, 1]
  end

  def repo_valid?
    silent_command("fossil", args: ["branch", "-R", cached_location]).success?
  end

  private

  def env
    { "PATH" => PATH.new(Formula["fossil"].opt_bin, ENV["PATH"]) }
  end

  sig { returns(String) }
  def cache_tag
    "fossil"
  end

  def clone_repo
    silent_command!("fossil", args: ["clone", @url, cached_location])
  end

  def update
    silent_command!("fossil", args: ["pull", "-R", cached_location])
  end
end

# Helper class for detecting a download strategy from a URL.
#
# @api private
class DownloadStrategyDetector
  def self.detect(url, using = nil)
    if using.nil?
      detect_from_url(url)
    elsif using.is_a?(Class) && using < AbstractDownloadStrategy
      using
    elsif using.is_a?(Symbol)
      detect_from_symbol(using)
    else
      raise TypeError,
            "Unknown download strategy specification #{using.inspect}"
    end
  end

  def self.detect_from_url(url)
    case url
    when %r{^https?://github\.com/[^/]+/[^/]+\.git$}
      GitHubGitDownloadStrategy
    when %r{^https?://.+\.git$},
         %r{^git://},
         %r{^https?://git\.sr\.ht/[^/]+/[^/]+$}
      GitDownloadStrategy
    when %r{^https?://www\.apache\.org/dyn/closer\.cgi},
         %r{^https?://www\.apache\.org/dyn/closer\.lua}
      CurlApacheMirrorDownloadStrategy
    when %r{^https?://(.+?\.)?googlecode\.com/svn},
         %r{^https?://svn\.},
         %r{^svn://},
         %r{^svn\+http://},
         %r{^http://svn\.apache\.org/repos/},
         %r{^https?://(.+?\.)?sourceforge\.net/svnroot/}
      SubversionDownloadStrategy
    when %r{^cvs://}
      CVSDownloadStrategy
    when %r{^hg://},
         %r{^https?://(.+?\.)?googlecode\.com/hg},
         %r{^https?://(.+?\.)?sourceforge\.net/hgweb/}
      MercurialDownloadStrategy
    when %r{^bzr://}
      BazaarDownloadStrategy
    when %r{^fossil://}
      FossilDownloadStrategy
    else
      CurlDownloadStrategy
    end
  end

  def self.detect_from_symbol(symbol)
    case symbol
    when :hg                     then MercurialDownloadStrategy
    when :nounzip                then NoUnzipCurlDownloadStrategy
    when :git                    then GitDownloadStrategy
    when :bzr                    then BazaarDownloadStrategy
    when :svn                    then SubversionDownloadStrategy
    when :curl                   then CurlDownloadStrategy
    when :cvs                    then CVSDownloadStrategy
    when :post                   then CurlPostDownloadStrategy
    when :fossil                 then FossilDownloadStrategy
    else
      raise TypeError, "Unknown download strategy #{symbol} was requested."
    end
  end
end
