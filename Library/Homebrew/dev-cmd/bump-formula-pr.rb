# frozen_string_literal: true

require "formula"
require "cli/parser"
require "utils/pypi"

module Homebrew
  module_function

  def bump_formula_pr_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump-formula-pr` [<options>] [<formula>]

        Create a pull request to update <formula> with a new URL or a new tag.

        If a <URL> is specified, the <SHA-256> checksum of the new download should also
        be specified. A best effort to determine the <SHA-256> and <formula> name will
        be made if either or both values are not supplied by the user.

        If a <tag> is specified, the Git commit <revision> corresponding to that tag
        should also be specified. A best effort to determine the <revision> will be made
        if the value is not supplied by the user.

        If a <version> is specified, a best effort to determine the <URL> and <SHA-256> or
        the <tag> and <revision> will be made if both values are not supplied by the user.

        *Note:* this command cannot be used to transition a formula from a
        URL-and-SHA-256 style specification into a tag-and-revision style specification,
        nor vice versa. It must use whichever style specification the formula already uses.
      EOS
      switch "-n", "--dry-run",
             description: "Print what would be done rather than doing it."
      switch "--write",
             depends_on:  "--dry-run",
             description: "When passed along with `--dry-run`, perform a not-so-dry run by making the expected "\
                          "file modifications but not taking any Git actions."
      switch "--no-audit",
             description: "Don't run `brew audit` before opening the PR."
      switch "--strict",
             description: "Run `brew audit --strict` before opening the PR."
      switch "--no-browse",
             description: "Print the pull request URL instead of opening in a browser."
      switch "--no-fork",
             description: "Don't try to fork the repository."
      comma_array "--mirror",
                  description: "Use the specified <URL> as a mirror URL. If <URL> is a comma-separated list "\
                               "of URLs, multiple mirrors will be added."
      flag   "--version=",
             description: "Use the specified <version> to override the value parsed from the URL or tag. Note "\
                          "that `--version=0` can be used to delete an existing version override from a "\
                          "formula if it has become redundant."
      flag   "--message=",
             description: "Append <message> to the default pull request message."
      flag   "--url=",
             description: "Specify the <URL> for the new download. If a <URL> is specified, the <SHA-256> "\
                          "checksum of the new download should also be specified."
      flag   "--sha256=",
             depends_on:  "--url=",
             description: "Specify the <SHA-256> checksum of the new download."
      flag   "--tag=",
             description: "Specify the new git commit <tag> for the formula."
      flag   "--revision=",
             depends_on:  "--tag=",
             description: "Specify the new git commit <revision> corresponding to the specified <tag>."
      switch "-f", "--force",
             description: "Ignore duplicate open PRs. Remove all mirrors if --mirror= was not specified."

      conflicts "--no-audit", "--strict"
      conflicts "--url", "--tag"
      max_named 1
    end
  end

  def use_correct_linux_tap(formula, args:)
    if OS.linux? && formula.tap.core_tap?
      tap_full_name = formula.tap.full_name.gsub("linuxbrew", "homebrew")
      homebrew_core_url = "https://github.com/#{tap_full_name}"
      homebrew_core_remote = "homebrew"
      homebrew_core_branch = "master"
      origin_branch = "#{homebrew_core_remote}/#{homebrew_core_branch}"
      previous_branch = Utils.popen_read("git -C \"#{formula.tap.path}\" symbolic-ref -q --short HEAD").chomp
      previous_branch = "master" if previous_branch.empty?
      formula_path = formula.path.to_s[%r{(Formula/.*)}, 1]

      if args.dry_run?
        ohai "git remote add #{homebrew_core_remote} #{homebrew_core_url}"
        ohai "git fetch #{homebrew_core_remote} #{homebrew_core_branch}"
        ohai "git cat-file -e #{origin_branch}:#{formula_path}"
        ohai "git checkout #{origin_branch}"
        return tap_full_name, origin_branch, previous_branch
      else
        formula.path.parent.cd do
          unless Utils.popen_read("git remote -v").match?(%r{^homebrew.*Homebrew/homebrew-core.*$})
            ohai "Adding #{homebrew_core_remote} remote"
            safe_system "git", "remote", "add", homebrew_core_remote, homebrew_core_url
          end
          ohai "Fetching #{origin_branch}"
          safe_system "git", "fetch", homebrew_core_remote, homebrew_core_branch
          if quiet_system "git", "cat-file", "-e", "#{origin_branch}:#{formula_path}"
            ohai "#{formula.full_name} exists in #{origin_branch}"
            safe_system "git", "checkout", origin_branch
            return tap_full_name, origin_branch, previous_branch
          end
        end
      end
    end
    origin_branch = Utils.popen_read("git", "-C", formula.tap.path.to_s, "symbolic-ref", "-q", "--short",
                                     "refs/remotes/origin/HEAD").chomp.presence
    origin_branch ||= "origin/master"
    [formula.tap&.full_name, origin_branch, "-"]
  end

  def bump_formula_pr
    args = bump_formula_pr_args.parse

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    # Use the user's browser, too.
    ENV["BROWSER"] = Homebrew::EnvConfig.browser

    formula = args.formulae.first

    new_url = args.url
    formula ||= determine_formula_from_url(new_url) if new_url
    raise FormulaUnspecifiedError unless formula

    tap_full_name, origin_branch, previous_branch = use_correct_linux_tap(formula, args: args)
    check_open_pull_requests(formula, tap_full_name, args: args)

    new_version = args.version
    check_closed_pull_requests(formula, tap_full_name, version: new_version, args: args) if new_version

    requested_spec = :stable
    formula_spec = formula.stable
    odie "#{formula}: no #{requested_spec} specification found!" unless formula_spec

    hash_type, old_hash = if (checksum = formula_spec.checksum)
      [checksum.hash_type, checksum.hexdigest]
    end

    new_hash = args[hash_type] if hash_type
    new_tag = args.tag
    new_revision = args.revision
    new_mirrors ||= args.mirror
    new_mirror ||= case new_url
    when %r{.*ftp.gnu.org/gnu.*}
      new_url.sub "ftp.gnu.org/gnu", "ftpmirror.gnu.org"
    when %r{.*download.savannah.gnu.org/*}
      new_url.sub "download.savannah.gnu.org", "download-mirror.savannah.gnu.org"
    when %r{.*www.apache.org/dyn/closer.lua\?path=.*}
      new_url.sub "www.apache.org/dyn/closer.lua?path=", "archive.apache.org/dist/"
    when %r{.*mirrors.ocf.berkeley.edu/debian.*}
      new_url.sub "mirrors.ocf.berkeley.edu/debian", "mirrorservice.org/sites/ftp.debian.org/debian"
    end
    new_mirrors ||= [new_mirror] unless new_mirror.nil?
    old_url = formula_spec.url
    old_tag = formula_spec.specs[:tag]
    old_formula_version = formula_version(formula, requested_spec)
    old_version = old_formula_version.to_s
    forced_version = new_version.present?
    new_url_hash = if new_url && new_hash
      check_closed_pull_requests(formula, tap_full_name, url: new_url, args: args) unless new_version
      true
    elsif new_tag && new_revision
      check_closed_pull_requests(formula, tap_full_name, url: old_url, tag: new_tag, args: args) unless new_version
      false
    elsif !hash_type
      odie "#{formula}: no --tag= or --version= argument specified!" if !new_tag && !new_version
      new_tag ||= old_tag.gsub(old_version, new_version)
      if new_tag == old_tag
        odie <<~EOS
          You need to bump this formula manually since the new tag
          and old tag are both #{new_tag}.
        EOS
      end
      check_closed_pull_requests(formula, tap_full_name, url: old_url, tag: new_tag, args: args) unless new_version
      resource_path, forced_version = fetch_resource(formula, new_version, old_url, tag: new_tag)
      new_revision = Utils.popen_read("git -C \"#{resource_path}\" rev-parse -q --verify HEAD")
      new_revision = new_revision.strip
      false
    elsif !new_url && !new_version
      odie "#{formula}: no --url= or --version= argument specified!"
    else
      new_url ||= PyPI.update_pypi_url(old_url, new_version)
      new_url ||= old_url.gsub(old_version, new_version)
      if new_url == old_url
        odie <<~EOS
          You need to bump this formula manually since the new URL
          and old URL are both:
            #{new_url}
        EOS
      end
      check_closed_pull_requests(formula, tap_full_name, url: new_url, args: args) unless new_version
      resource_path, forced_version = fetch_resource(formula, new_version, new_url)
      tar_file_extensions = %w[.tar .tb2 .tbz .tbz2 .tgz .tlz .txz .tZ]
      if tar_file_extensions.any? { |extension| new_url.include? extension }
        gnu_tar_gtar_path = HOMEBREW_PREFIX/"opt/gnu-tar/bin/gtar"
        gnu_tar_gtar = gnu_tar_gtar_path if gnu_tar_gtar_path.executable?
        tar = which("gtar") || gnu_tar_gtar || which("tar")
        if Utils.popen_read(tar, "-tf", resource_path).match?(%r{/.*\.})
          new_hash = resource_path.sha256
        else
          odie "#{resource_path} is not a valid tar file!"
        end
      else
        new_hash = resource_path.sha256
      end
    end

    replacement_pairs = []
    if requested_spec == :stable && formula.revision.nonzero?
      replacement_pairs << [
        /^  revision \d+\n(\n(  head "))?/m,
        "\\2",
      ]
    end

    replacement_pairs += formula_spec.mirrors.map do |mirror|
      [
        / +mirror "#{Regexp.escape(mirror)}"\n/m,
        "",
      ]
    end

    replacement_pairs += if new_url_hash
      [
        [
          /#{Regexp.escape(formula_spec.url)}/,
          new_url,
        ],
        [
          old_hash,
          new_hash,
        ],
      ]
    else
      [
        [
          formula_spec.specs[:tag],
          new_tag,
        ],
        [
          formula_spec.specs[:revision],
          new_revision,
        ],
      ]
    end

    read_only_run = args.dry_run? && !args.write?
    old_contents = File.read(formula.path) unless read_only_run

    if new_mirrors
      replacement_pairs << [
        /^( +)(url "#{Regexp.escape(new_url)}"\n)/m,
        "\\1\\2\\1mirror \"#{new_mirrors.join("\"\n\\1mirror \"")}\"\n",
      ]
    end

    # When bumping a linux-only formula, one needs to also delete the
    # sha256 linux bottle line if it exists. That's because of running
    # test-bot with --keep-old option in linuxbrew-core.
    formula_contents = formula.path.read
    if formula_contents.include?("depends_on :linux") && formula_contents.include?("=> :x86_64_linux")
      replacement_pairs << [
        /^    sha256 ".+" => :x86_64_linux\n/m,
        "\\2",
      ]
    end

    if forced_version && new_version != "0"
      replacement_pairs << if File.read(formula.path).include?("version \"#{old_formula_version}\"")
        [
          old_formula_version.to_s,
          new_version,
        ]
      elsif new_mirrors
        [
          /^( +)(mirror "#{Regexp.escape(new_mirrors.last)}"\n)/m,
          "\\1\\2\\1version \"#{new_version}\"\n",
        ]
      elsif new_url
        [
          /^( +)(url "#{Regexp.escape(new_url)}"\n)/m,
          "\\1\\2\\1version \"#{new_version}\"\n",
        ]
      elsif new_revision
        [
          /^( {2})( +)(:revision => "#{new_revision}"\n)/m,
          "\\1\\2\\3\\1version \"#{new_version}\"\n",
        ]
      end
    elsif forced_version && new_version == "0"
      replacement_pairs << [
        /^  version "[\w.\-+]+"\n/m,
        "",
      ]
    end
    new_contents = inreplace_pairs(formula.path, replacement_pairs.uniq.compact, args: args)

    new_formula_version = formula_version(formula, requested_spec, new_contents)

    if !new_mirrors && !formula_spec.mirrors.empty?
      if args.force?
        opoo "#{formula}: Removing all mirrors because a --mirror= argument was not specified."
      else
        odie <<~EOS
          #{formula}: a --mirror= argument for updating the mirror URL was not specified.
          Use --force to remove all mirrors.
        EOS
      end
    end

    if new_formula_version < old_formula_version
      formula.path.atomic_write(old_contents) unless read_only_run
      odie <<~EOS
        You need to bump this formula manually since changing the
        version from #{old_formula_version} to #{new_formula_version} would be a downgrade.
      EOS
    elsif new_formula_version == old_formula_version
      formula.path.atomic_write(old_contents) unless read_only_run
      odie <<~EOS
        You need to bump this formula manually since the new version
        and old version are both #{new_formula_version}.
      EOS
    end

    alias_rename = alias_update_pair(formula, new_formula_version)
    if alias_rename.present?
      ohai "renaming alias #{alias_rename.first} to #{alias_rename.last}"
      alias_rename.map! { |a| formula.tap.alias_dir/a }
    end

    ohai "brew update-python-resources #{formula.name}"
    unless read_only_run
      PyPI.update_python_resources! formula, new_formula_version, silent: true, ignore_non_pypi_packages: true
    end

    run_audit(formula, alias_rename, old_contents, args: args)

    formula.path.parent.cd do
      _, base_branch = origin_branch.split("/")
      branch = "bump-#{formula.name}-#{new_formula_version}"
      git_dir = Utils.popen_read("git rev-parse --git-dir").chomp
      shallow = !git_dir.empty? && File.exist?("#{git_dir}/shallow")
      changed_files = [formula.path]
      changed_files += alias_rename if alias_rename.present?

      if args.dry_run?
        ohai "try to fork repository with GitHub API" unless args.no_fork?
        ohai "git fetch --unshallow origin" if shallow
        ohai "git add #{alias_rename.first} #{alias_rename.last}" if alias_rename.present?
        ohai "git checkout --no-track -b #{branch} #{origin_branch}"
        ohai "git commit --no-edit --verbose --message='#{formula.name} " \
             "#{new_formula_version}' -- #{changed_files.join(" ")}"
        ohai "git push --set-upstream $HUB_REMOTE #{branch}:#{branch}"
        ohai "git checkout --quiet #{previous_branch}"
        ohai "create pull request with GitHub API (base branch: #{base_branch})"
      else

        if args.no_fork?
          remote_url = Utils.popen_read("git remote get-url --push origin").chomp
          username = formula.tap.user
        else
          remote_url, username = forked_repo_info(formula, tap_full_name, old_contents)
        end

        safe_system "git", "fetch", "--unshallow", "origin" if shallow
        safe_system "git", "add", *alias_rename if alias_rename.present?
        safe_system "git", "checkout", "--no-track", "-b", branch, origin_branch
        safe_system "git", "commit", "--no-edit", "--verbose",
                    "--message=#{formula.name} #{new_formula_version}",
                    "--", *changed_files
        safe_system "git", "push", "--set-upstream", remote_url, "#{branch}:#{branch}"
        safe_system "git", "checkout", "--quiet", previous_branch
        pr_message = <<~EOS
          Created with `brew bump-formula-pr`.
        EOS
        user_message = args.message
        if user_message
          pr_message += "\n" + <<~EOS
            ---

            #{user_message}
          EOS
        end
        pr_title = "#{formula.name} #{new_formula_version}"

        begin
          url = GitHub.create_pull_request(tap_full_name, pr_title,
                                           "#{username}:#{branch}", base_branch, pr_message)["html_url"]
          if args.no_browse?
            puts url
          else
            exec_browser url
          end
        rescue *GitHub.api_errors => e
          odie "Unable to open pull request: #{e.message}!"
        end
      end
    end
  end

  def determine_formula_from_url(url)
    # Split the new URL on / and find any formulae that have the same URL
    # except for the last component, but don't try to match any more than the
    # first five components since sometimes the last component isn't the only
    # one to change.
    url_split = url.split("/")
    maximum_url_components_to_match = 5
    components_to_match = [url_split.count - 1, maximum_url_components_to_match].min
    base_url = url_split.first(components_to_match).join("/")
    base_url = /#{Regexp.escape(base_url)}/
    guesses = []
    Formula.each do |f|
      guesses << f if f.stable&.url && f.stable.url.match(base_url)
    end
    return guesses.shift if guesses.count == 1
    return if guesses.count <= 1

    odie "Couldn't guess formula for sure; could be one of these:\n#{guesses.map(&:name).join(", ")}"
  end

  def fetch_resource(formula, new_version, url, **specs)
    resource = Resource.new
    resource.url(url, specs)
    resource.owner = Resource.new(formula.name)
    forced_version = new_version && new_version != resource.version
    resource.version = new_version if forced_version
    odie "No --version= argument specified!" unless resource.version
    [resource.fetch, forced_version]
  end

  def forked_repo_info(formula, tap_full_name, old_contents)
    response = GitHub.create_fork(tap_full_name)
  rescue GitHub::AuthenticationFailedError, *GitHub.api_errors => e
    formula.path.atomic_write(old_contents)
    odie "Unable to fork: #{e.message}!"
  else
    # GitHub API responds immediately but fork takes a few seconds to be ready.
    sleep 1 until GitHub.check_fork_exists(tap_full_name)
    remote_url = if system("git", "config", "--local", "--get-regexp", "remote\..*\.url", "git@github.com:.*")
      response.fetch("ssh_url")
    else
      url = response.fetch("clone_url")
      if (api_token = Homebrew::EnvConfig.github_api_token)
        url.gsub!(%r{^https://github\.com/}, "https://#{api_token}@github.com/")
      end
      url
    end
    username = response.fetch("owner").fetch("login")
    [remote_url, username]
  end

  def inreplace_pairs(path, replacement_pairs, args:)
    read_only_run = args.dry_run? && !args.write?
    if read_only_run
      str = path.open("r") { |f| Formulary.ensure_utf8_encoding(f).read }
      contents = StringInreplaceExtension.new(str)
      replacement_pairs.each do |old, new|
        ohai "replace #{old.inspect} with #{new.inspect}" unless args.quiet?
        raise "No old value for new value #{new}! Did you pass the wrong arguments?" unless old

        contents.gsub!(old, new)
      end
      raise Utils::InreplaceError, path => contents.errors unless contents.errors.empty?

      path.atomic_write(contents.inreplace_string) if args.write?
      contents.inreplace_string
    else
      Utils::Inreplace.inreplace(path) do |s|
        replacement_pairs.each do |old, new|
          ohai "replace #{old.inspect} with #{new.inspect}" unless args.quiet?
          raise "No old value for new value #{new}! Did you pass the wrong arguments?" unless old

          s.gsub!(old, new)
        end
      end
      path.open("r") { |f| Formulary.ensure_utf8_encoding(f).read }
    end
  end

  def formula_version(formula, spec, contents = nil)
    name = formula.name
    path = formula.path
    if contents
      Formulary.from_contents(name, path, contents, spec).version
    else
      Formulary::FormulaLoader.new(name, path).get_formula(spec).version
    end
  end

  def fetch_pull_requests(query, tap_full_name, state: nil)
    GitHub.issues_for_formula(query, tap_full_name: tap_full_name, state: state).select do |pr|
      pr["html_url"].include?("/pull/") &&
        /(^|\s)#{Regexp.quote(query)}(:|\s|$)/i =~ pr["title"]
    end
  rescue GitHub::RateLimitExceededError => e
    opoo e.message
    []
  end

  def check_open_pull_requests(formula, tap_full_name, args:)
    # check for open requests
    pull_requests = fetch_pull_requests(formula.name, tap_full_name, state: "open")
    check_for_duplicate_pull_requests(pull_requests, args: args)
  end

  def check_closed_pull_requests(formula, tap_full_name, version: nil, url: nil, tag: nil, args:)
    unless version
      specs = {}
      specs[:tag] = tag if tag
      version = Version.detect(url, specs)
    end
    # if we haven't already found open requests, try for an exact match across closed requests
    pull_requests = fetch_pull_requests("#{formula.name} #{version}", tap_full_name, state: "closed")
    check_for_duplicate_pull_requests(pull_requests, args: args)
  end

  def check_for_duplicate_pull_requests(pull_requests, args:)
    return if pull_requests.blank?

    duplicates_message = <<~EOS
      These pull requests may be duplicates:
      #{pull_requests.map { |pr| "#{pr["title"]} #{pr["html_url"]}" }.join("\n")}
    EOS
    error_message = "Duplicate PRs should not be opened. Use --force to override this error."
    if args.force? && !args.quiet?
      opoo duplicates_message
    elsif !args.force? && args.quiet?
      odie error_message
    elsif !args.force?
      odie <<~EOS
        #{duplicates_message.chomp}
        #{error_message}
      EOS
    end
  end

  def alias_update_pair(formula, new_formula_version)
    versioned_alias = formula.aliases.grep(/^.*@\d+(\.\d+)?$/).first
    return if versioned_alias.nil?

    name, old_alias_version = versioned_alias.split("@")
    new_alias_regex = (old_alias_version.split(".").length == 1) ? /^\d+/ : /^\d+\.\d+/
    new_alias_version, = *new_formula_version.to_s.match(new_alias_regex)
    return if Version.create(new_alias_version) <= Version.create(old_alias_version)

    [versioned_alias, "#{name}@#{new_alias_version}"]
  end

  def run_audit(formula, alias_rename, old_contents, args:)
    if args.dry_run?
      if args.no_audit?
        ohai "Skipping `brew audit`"
      elsif args.strict?
        ohai "brew audit --strict #{formula.path.basename}"
      else
        ohai "brew audit #{formula.path.basename}"
      end
      return
    end
    FileUtils.mv alias_rename.first, alias_rename.last if alias_rename.present?
    failed_audit = false
    if args.no_audit?
      ohai "Skipping `brew audit`"
    elsif args.strict?
      system HOMEBREW_BREW_FILE, "audit", "--strict", formula.path
      failed_audit = !$CHILD_STATUS.success?
    else
      system HOMEBREW_BREW_FILE, "audit", formula.path
      failed_audit = !$CHILD_STATUS.success?
    end
    return unless failed_audit

    formula.path.atomic_write(old_contents)
    FileUtils.mv alias_rename.last, alias_rename.first if alias_rename.present?
    odie "`brew audit` failed!"
  end
end
