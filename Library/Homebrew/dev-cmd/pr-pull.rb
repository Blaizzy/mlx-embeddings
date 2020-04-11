# frozen_string_literal: true

require "cli/parser"
require "utils/github"
require "tmpdir"
require "bintray"

module Homebrew
  module_function

  def pr_pull_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pr-pull` <pull_request>

        Download and publish bottles, and apply the bottle commit from a
        pull request with artifacts generated from GitHub Actions.
        Requires write access to the repository.
      EOS
      switch "--no-publish",
             description: "Download the bottles, apply the bottle commit, and "\
                          "upload the bottles to Bintray, but don't publish them."
      switch "--no-upload",
             description: "Download the bottles and apply the bottle commit, "\
                          "but don't upload to Bintray."
      switch "--dry-run", "-n",
             description: "Print what would be done rather than doing it."
      switch "--clean",
             description: "Do not amend the commits from pull requests."
      switch "--branch-okay",
             description: "Do not warn if pulling to a branch besides master (useful for testing)."
      switch "--resolve",
             description: "When a patch fails to apply, leave in progress and allow user to resolve, instead "\
                          "of aborting."
      flag "--workflow=",
           description: "Retrieve artifacts from the specified workflow (default: tests.yml)."
      flag "--artifact=",
           description: "Download artifacts with the specified name (default: bottles)."
      flag "--bintray-org=",
           description: "Upload to the specified Bintray organisation (default: homebrew)."
      flag "--tap=",
           description: "Target repository tap (default: homebrew/core)."
      switch :verbose
      switch :debug
      min_named 1
    end
  end

  def setup_git_environment!
    # Passthrough Git environment variables
    ENV["GIT_COMMITTER_NAME"] = ENV["HOMEBREW_GIT_NAME"] if ENV["HOMEBREW_GIT_NAME"]
    ENV["GIT_COMMITTER_EMAIL"] = ENV["HOMEBREW_GIT_EMAIL"] if ENV["HOMEBREW_GIT_EMAIL"]

    # Depending on user configuration, git may try to invoke gpg.
    return unless Utils.popen_read("git config --get --bool commit.gpgsign").chomp == "true"

    begin
      gnupg = Formula["gnupg"]
    rescue FormulaUnavailableError
      nil
    else
      if gnupg.installed?
        path = PATH.new(ENV.fetch("PATH"))
        path.prepend(gnupg.installed_prefix/"bin")
        ENV["PATH"] = path
      end
    end
  end

  def signoff!(pr, path: ".")
    message = Utils.popen_read "git", "-C", path, "log", "-1", "--pretty=%B"
    close_message = "Closes ##{pr}."
    message += "\n#{close_message}" unless message.include? close_message
    if Homebrew.args.dry_run?
      puts "git commit --amend --signoff -m $message"
    else
      safe_system "git", "-C", path, "commit", "--amend", "--signoff", "--allow-empty", "-q", "-m", message
    end
  end

  def cherry_pick_pr!(pr, path: ".")
    if Homebrew.args.dry_run?
      puts <<~EOS
        git fetch --force origin +refs/pull/#{pr}/head
        git merge-base HEAD FETCH_HEAD
        git cherry-pick --ff --allow-empty $merge_base..FETCH_HEAD
      EOS
    else
      safe_system "git", "-C", path, "fetch", "--quiet", "--force", "origin", "+refs/pull/#{pr}/head"
      merge_base = Utils.popen_read("git", "-C", path, "merge-base", "HEAD", "FETCH_HEAD").strip
      commit_count = Utils.popen_read("git", "-C", path, "rev-list", "#{merge_base}..FETCH_HEAD").lines.count

      # git cherry-pick unfortunately has no quiet option
      ohai "Cherry-picking #{commit_count} commit#{"s" unless commit_count == 1} from ##{pr}"
      cherry_pick_args = "git", "-C", path, "cherry-pick", "--ff", "--allow-empty", "#{merge_base}..FETCH_HEAD"
      result = Homebrew.args.verbose? ? system(*cherry_pick_args) : quiet_system(*cherry_pick_args)

      unless result
        if Homebrew.args.resolve?
          odie "Cherry-pick failed: try to resolve it."
        else
          system "git", "-C", path, "cherry-pick", "--abort"
          odie "Cherry-pick failed!"
        end
      end
    end
  end

  def check_branch(path, ref)
    branch = Utils.popen_read("git", "-C", path, "symbolic-ref", "--short", "HEAD").strip

    return if branch == ref || args.clean? || args.branch_okay?

    opoo "Current branch is #{branch}: do you need to pull inside #{ref}?"
  end

  def pr_pull
    pr_pull_args.parse

    bintray_user = ENV["HOMEBREW_BINTRAY_USER"]
    bintray_key = ENV["HOMEBREW_BINTRAY_KEY"]
    bintray_org = args.bintray_org || "homebrew"

    if bintray_user.blank? || bintray_key.blank?
      odie "Missing HOMEBREW_BINTRAY_USER or HOMEBREW_BINTRAY_KEY variables!" if !args.dry_run? && !args.no_upload?
    else
      bintray = Bintray.new(user: bintray_user, key: bintray_key, org: bintray_org)
    end

    workflow = args.workflow || "tests.yml"
    artifact = args.artifact || "bottles"
    tap = Tap.fetch(args.tap || "homebrew/core")

    setup_git_environment!

    args.named.each do |arg|
      arg = "#{tap.default_remote}/pull/#{arg}" if arg.to_i.positive?
      url_match = arg.match HOMEBREW_PULL_OR_COMMIT_URL_REGEX
      _, user, repo, pr = *url_match
      odie "Not a GitHub pull request: #{arg}" unless pr

      check_branch tap.path, "master"

      ohai "Fetching #{tap} pull request ##{pr}"
      Dir.mktmpdir pr do |dir|
        cd dir do
          GitHub.fetch_artifact(user, repo, pr, dir, workflow_id: workflow, artifact_name: artifact)
          cherry_pick_pr! pr, path: tap.path
          signoff! pr, path: tap.path unless args.clean?

          if Homebrew.args.dry_run?
            puts "brew bottle --merge --write #{Dir["*.json"].join " "}"
          else
            quiet_system "#{HOMEBREW_PREFIX}/bin/brew", "bottle", "--merge", "--write", *Dir["*.json"]
          end

          next if args.no_upload?

          if Homebrew.args.dry_run?
            puts "Upload bottles described by these JSON files to Bintray:\n  #{Dir["*.json"].join("\n  ")}"
          else
            bintray.upload_bottle_json Dir["*.json"], publish_package: !args.no_publish?
          end
        end
      end
    end
  end
end
