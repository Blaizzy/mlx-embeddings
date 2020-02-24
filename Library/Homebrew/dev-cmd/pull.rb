# frozen_string_literal: true

require "net/http"
require "net/https"
require "json"
require "cli/parser"
require "formula"
require "formulary"
require "version"
require "pkg_version"
require "bottle_publisher"
require "formula_info"

module GitHub
  module_function

  # Return the corresponding test-bot user name for the given GitHub organization.
  def test_bot_user(user, test_bot)
    return test_bot if test_bot
    return "BrewTestBot" if user.casecmp("homebrew").zero?

    "#{user.capitalize}TestBot"
  end
end

module Homebrew
  module_function

  def pull_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pull` [<options>] <patch>

        Get a patch from a GitHub commit or pull request and apply it to Homebrew.
        Optionally, publish updated bottles for any formulae changed by the patch.

        Each <patch> may be the number of a pull request in `homebrew/core`, the URL of any pull request
        or commit on GitHub or a "https://jenkins.brew.sh/job/..." testing job URL.
      EOS
      switch "--bottle",
             description: "Handle bottles, pulling the bottle-update commit and publishing files on Bintray."
      switch "--bump",
             description: "For one-formula PRs, automatically reword commit message to our preferred format."
      switch "--clean",
             description: "Do not rewrite or otherwise modify the commits found in the pulled PR."
      switch "--ignore-whitespace",
             description: "Silently ignore whitespace discrepancies when applying diffs."
      switch "--resolve",
             description: "When a patch fails to apply, leave in progress and allow user to resolve, instead "\
                          "of aborting."
      switch "--branch-okay",
             description: "Do not warn if pulling to a branch besides master (useful for testing)."
      switch "--no-pbcopy",
             description: "Do not copy anything to the system clipboard."
      switch "--no-publish",
             description: "Do not publish bottles to Bintray."
      switch "--warn-on-publish-failure",
             description: "Do not exit if there's a failure publishing bottles on Bintray."
      flag   "--bintray-org=",
             description: "Publish bottles to the specified Bintray <organisation>."
      flag   "--test-bot-user=",
             description: "Pull the bottle block commit from the specified <user> on GitHub."
      switch :verbose
      switch :debug
    end
  end

  def pull
    odie "You meant `git pull --rebase`." if ARGV[0] == "--rebase"

    pull_args.parse

    if ARGV.named.empty?
      raise UsageError, "This command requires at least one argument containing a URL or pull request number"
    end

    # Passthrough Git environment variables for e.g. git am
    ENV["GIT_COMMITTER_NAME"] = ENV["HOMEBREW_GIT_NAME"] if ENV["HOMEBREW_GIT_NAME"]
    ENV["GIT_COMMITTER_EMAIL"] = ENV["HOMEBREW_GIT_EMAIL"] if ENV["HOMEBREW_GIT_EMAIL"]

    # Depending on user configuration, git may try to invoke gpg.
    if Utils.popen_read("git config --get --bool commit.gpgsign").chomp == "true"
      begin
        gnupg = Formula["gnupg"]
      rescue FormulaUnavailableError # rubocop:disable Lint/SuppressedException
      else
        if gnupg.installed?
          path = PATH.new(ENV.fetch("PATH"))
          path.prepend(gnupg.installed_prefix/"bin")
          ENV["PATH"] = path
        end
      end
    end

    do_bump = args.bump? && !args.clean?

    tap = nil

    ARGV.named.each do |arg|
      arg = "#{CoreTap.instance.default_remote}/pull/#{arg}" if arg.to_i.positive?
      if (testing_match = arg.match %r{/job/Homebrew.*Testing/(\d+)})
        tap = ARGV.value("tap")
        tap = if tap&.start_with?("homebrew/")
          Tap.fetch("homebrew", tap.delete_prefix("homebrew/"))
        elsif tap
          odie "Tap option did not start with \"homebrew/\": #{tap}"
        else
          CoreTap.instance
        end
        _, testing_job = *testing_match
        url = "https://github.com/Homebrew/homebrew-#{tap.repo}/compare/master...BrewTestBot:testing-#{testing_job}"
        odie "--bottle is required for testing job URLs!" unless args.bottle?
      elsif (api_match = arg.match HOMEBREW_PULL_API_REGEX)
        _, user, repo, issue = *api_match
        url = "https://github.com/#{user}/#{repo}/pull/#{issue}"
        tap = Tap.fetch(user, repo) if repo.match?(HOMEBREW_OFFICIAL_REPO_PREFIXES_REGEX)
      elsif (url_match = arg.match HOMEBREW_PULL_OR_COMMIT_URL_REGEX)
        url, user, repo, issue = *url_match
        tap = Tap.fetch(user, repo) if repo.match?(HOMEBREW_OFFICIAL_REPO_PREFIXES_REGEX)
      else
        odie "Not a GitHub pull request or commit: #{arg}"
      end

      odie "No pull request detected!" if !testing_job && args.bottle? && issue.nil?

      if tap
        tap.install unless tap.installed?
        Dir.chdir tap.path
      else
        Dir.chdir HOMEBREW_REPOSITORY
      end

      # The cache directory seems like a good place to put patches.
      HOMEBREW_CACHE.mkpath

      # Store current revision and branch
      merge_commit = merge_commit?(url)
      orig_revision = `git rev-parse --short HEAD`.strip
      branch = `git symbolic-ref --short HEAD`.strip

      if branch != "master" && !args.clean? && !args.branch_okay?
        opoo "Current branch is #{branch}: do you need to pull inside master?"
      end

      unless merge_commit
        patch_puller = PatchPuller.new(url, args)
        patch_puller.fetch_patch
        patch_changes = files_changed_in_patch(patch_puller.patchpath, tap)

        is_bumpable = patch_changes[:formulae].length == 1 && patch_changes[:others].empty?
        check_bumps(patch_changes) if do_bump
        old_versions = current_versions_from_info_external(patch_changes[:formulae].first) if is_bumpable
        patch_puller.apply_patch
      end

      end_revision = head_revision(url, merge_commit)

      changed_formulae_names = []

      if tap
        Utils.popen_read(
          "git", "diff-tree", "-r", "--name-only",
          "--diff-filter=AM", orig_revision, end_revision, "--", tap.formula_dir.to_s
        ).each_line do |line|
          next unless line.end_with? ".rb\n"

          name = "#{tap.name}/#{File.basename(line.chomp, ".rb")}"
          changed_formulae_names << name
        end
      end

      fetch_bottles = false
      changed_formulae_names.each do |name|
        next if ENV["HOMEBREW_DISABLE_LOAD_FORMULA"]

        begin
          f = Formula[name]
        rescue Exception # rubocop:disable Lint/RescueException
          # Make sure we catch syntax errors.
          next
        end

        if f.stable
          stable_urls = [f.stable.url] + f.stable.mirrors
          stable_urls.grep(%r{^https://dl.bintray.com/homebrew/mirror/}) do |mirror_url|
            check_bintray_mirror(f.full_name, mirror_url)
          end
        end

        if args.bottle?
          if f.bottle_unneeded?
            ohai "#{f}: skipping unneeded bottle."
          elsif f.bottle_disabled?
            ohai "#{f}: skipping disabled bottle: #{f.bottle_disable_reason}"
          else
            fetch_bottles = true
          end
        else
          next unless f.bottle_defined?

          opoo "#{f.full_name} has a bottle: do you need to update it with --bottle?"
        end
      end

      orig_message = message = `git log HEAD^.. --format=%B`
      if issue && !args.clean?
        ohai "Patch closes issue ##{issue}"
        close_message = "Closes ##{issue}."
        # If this is a pull request, append a close message.
        message += "\n#{close_message}" unless message.include? close_message
      end

      if changed_formulae_names.empty?
        odie "Cannot bump: no changed formulae found after applying patch" if do_bump
        is_bumpable = false
      end

      is_bumpable = false if args.clean?
      is_bumpable = false if ENV["HOMEBREW_DISABLE_LOAD_FORMULA"]

      if is_bumpable
        formula = Formula[changed_formulae_names.first]
        new_versions = current_versions_from_info_external(patch_changes[:formulae].first)
        orig_subject = message.empty? ? "" : message.lines.first.chomp
        bump_subject = subject_for_bump(formula, old_versions, new_versions)
        if do_bump
          odie "No version changes found for #{formula.name}" if bump_subject.nil?
          unless orig_subject == bump_subject
            ohai "New bump commit subject: #{bump_subject}"
            pbcopy bump_subject unless args.no_pbcopy?
            message = "#{bump_subject}\n\n#{message}"
          end
        elsif bump_subject != orig_subject && !bump_subject.nil?
          opoo "Nonstandard bump subject: #{orig_subject}"
          opoo "Subject should be: #{bump_subject}"
        end
      end

      if message != orig_message && !args.clean?
        safe_system "git", "commit", "--amend", "--signoff", "--allow-empty", "-q", "-m", message
      end

      if fetch_bottles
        bottle_commit_url = if testing_job
          bottle_branch = "testing-bottle-#{testing_job}"
          url
        else
          bottle_branch = "pull-bottle-#{issue}"
          bot_username = GitHub.test_bot_user(user, args.test_bot_user)
          "https://github.com/#{bot_username}/homebrew-#{tap.repo}/compare/#{user}:master...pr-#{issue}"
        end

        curl "--silent", "--fail", "--output", "/dev/null", "--head", bottle_commit_url

        if merge_commit
          fetch_merge_patch(bottle_commit_url, args, issue)
        else
          fetch_bottles_patch(bottle_commit_url, args, bottle_branch, branch, orig_revision)
        end
        BottlePublisher.new(
          tap,
          changed_formulae_names,
          args.bintray_org,
          args.no_publish?,
          args.warn_on_publish_failure?,
        ).publish_and_check_bottles
      elsif merge_commit
        fetch_merge_patch(url, args, issue)
      end

      ohai "Patch changed:"
      safe_system "git", "diff-tree", "-r", "--stat", orig_revision, end_revision
    end
  end

  def check_bumps(patch_changes)
    if patch_changes[:formulae].empty?
      odie "No changed formulae found to bump"
    elsif patch_changes[:formulae].length > 1
      odie "Can only bump one changed formula; bumped #{patch_changes[:formulae]}"
    elsif !patch_changes[:others].empty?
      odie "Cannot bump if non-formula files are changed"
    end
  end

  def merge_commit?(url)
    pr_number = url[%r{/pull\/([0-9]+)}, 1]
    return false unless pr_number

    safe_system "git", "fetch", "--quiet", "origin", "pull/#{pr_number}/head"
    Utils.popen_read("git", "rev-list", "--parents", "-n1", "FETCH_HEAD").count(" ") > 1
  end

  def head_revision(_url, fetched)
    Utils.popen_read("git", "rev-parse", fetched ? "FETCH_HEAD" : "HEAD").strip
  end

  def fetch_bottles_patch(bottle_commit_url, args, bottle_branch, branch, orig_revision)
    safe_system "git", "checkout", "--quiet", "-B", bottle_branch, orig_revision
    PatchPuller.new(bottle_commit_url, args, "bottle commit").pull_patch
    safe_system "git", "rebase", "--quiet", branch
    safe_system "git", "checkout", "--quiet", branch
    safe_system "git", "merge", "--quiet", "--ff-only", "--no-edit", bottle_branch
    safe_system "git", "branch", "--quiet", "-D", bottle_branch
  end

  def fetch_merge_patch(url, args, issue)
    PatchPuller.new(url, args, "merge commit").pull_merge_commit(issue)
  end

  class PatchPuller
    attr_reader :base_url
    attr_reader :patch_url
    attr_reader :patchpath

    def initialize(url, args, description = nil)
      @base_url = url
      # GitHub provides commits/pull-requests raw patches using this URL.
      @patch_url = url + ".patch"
      @patchpath = HOMEBREW_CACHE + File.basename(patch_url)
      @description = description
      @args = args
    end

    def pull_patch
      fetch_patch
      apply_patch
    end

    def pull_merge_commit(issue)
      # Used by forks of homebrew-core that use merge-commits (for example linuxbrew)
      ohai "Fast-forwarding to the merge commit"
      test_bot_origin = patch_url[%r{(https://github\.com/[\w-]+/[\w-]+)/compare/}, 1]
      safe_system "git", "fetch", "--quiet", test_bot_origin, "pr-#{issue}" if test_bot_origin
      safe_system "git", "merge", "--quiet", "--ff-only", "--no-edit", "FETCH_HEAD"
      return if $CHILD_STATUS.success?

      safe_system "git", "reset", "--hard", "FETCH_HEAD"
      odie <<~EOS
        Not possible to fast-forward.
        Maybe somebody pushed commits to origin/master between the merge commit creation and now.
      EOS
    end

    def fetch_patch
      extra_msg = @description ? "(#{@description})" : nil
      ohai "Fetching patch #{extra_msg}"
      puts "Patch: #{patch_url}"
      curl_download patch_url, to: patchpath
    end

    def apply_patch
      # Applies a patch previously downloaded with fetch_patch()
      # Deletes the patch file as a side effect, regardless of success

      ohai "Applying patch"
      patch_args = []
      # Normally we don't want whitespace errors, but squashing them can break
      # patches so an option is provided to skip this step.
      if @args.ignore_whitespace? || @args.clean?
        patch_args << "--whitespace=nowarn"
      else
        patch_args << "--whitespace=fix"
      end

      # Fall back to three-way merge if patch does not apply cleanly
      patch_args << "-3"
      patch_args << patchpath

      begin
        safe_system "git", "am", *patch_args
      rescue ErrorDuringExecution
        if @args.resolve?
          odie "Patch failed to apply: try to resolve it."
        else
          system "git", "am", "--abort"
          odie "Patch failed to apply: aborted."
        end
      ensure
        patchpath.unlink
      end
    end
  end

  # List files changed by a patch, partitioned in to those that are (probably)
  # formula definitions, and those which aren't. Only applies to patches on
  # Homebrew core or taps, based simply on relative pathnames of affected files.
  def files_changed_in_patch(patchfile, tap)
    files = []
    formulae = []
    others = []
    File.foreach(patchfile) do |line|
      files << Regexp.last_match(1) if line =~ %r{^\+\+\+ b/(.*)}
    end
    files.each do |file|
      if tap&.formula_file?(file)
        formula_name = File.basename(file, ".rb")
        formulae << formula_name unless formulae.include?(formula_name)
      else
        others << file
      end
    end
    { files: files, formulae: formulae, others: others }
  end

  # Get current formula versions without loading formula definition in this process.
  # Returns info as a hash (type => version), for pull.rb's internal use.
  # Uses special key `:nonexistent => true` for nonexistent formulae.
  def current_versions_from_info_external(formula_name)
    info = FormulaInfo.lookup(formula_name)
    versions = {}
    if info
      [:stable, :devel, :head].each do |spec_type|
        versions[spec_type] = info.version(spec_type)
      end
    else
      versions[:nonexistent] = true
    end
    versions
  end

  def subject_for_bump(formula, old, new)
    if old[:nonexistent]
      # New formula
      headline_ver = if new[:stable]
        new[:stable]
      elsif new[:devel]
        new[:devel]
      else
        new[:head]
      end
      subject = "#{formula.name} #{headline_ver} (new formula)"
    else
      # Update to existing formula
      subject_strs = []
      formula_name_str = formula.name
      if old[:stable] != new[:stable]
        if new[:stable].nil?
          subject_strs << "remove stable"
          formula_name_str += ":" # just for cosmetics
        else
          subject_strs << new[:stable]
        end
      end
      if old[:devel] != new[:devel]
        if new[:devel].nil?
          # Only bother mentioning if there's no accompanying stable change
          if !new[:stable].nil? && old[:stable] == new[:stable]
            subject_strs << "remove devel"
            formula_name_str += ":" # just for cosmetics
          end
        else
          subject_strs << "#{new[:devel]} (devel)"
        end
      end
      subject = subject_strs.empty? ? nil : "#{formula_name_str} #{subject_strs.join(", ")}"
    end
    subject
  end

  def pbcopy(text)
    Utils.popen_write("pbcopy") { |io| io.write text }
  end

  def check_bintray_mirror(name, url)
    headers, = curl_output("--connect-timeout", "15", "--location", "--head", url)
    status_code = headers.scan(%r{^HTTP\/.* (\d+)}).last.first
    return if status_code.start_with?("2")

    opoo "The Bintray mirror #{url} is not reachable (HTTP status code #{status_code})."
    opoo "Do you need to upload it with `brew mirror #{name}`?"
  end
end
