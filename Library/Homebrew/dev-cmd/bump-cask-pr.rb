# frozen_string_literal: true

require "cask"
require "cli/parser"
require "utils/tar"

module Homebrew
  module_function

  def bump_cask_pr_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump-cask-pr` [<options>] [<cask>]

        Create a pull request to update <cask> with a new version.

        A best effort to determine the <SHA-256> will be made if the value is not
        supplied by the user.
      EOS
      switch "-n", "--dry-run",
             description: "Print what would be done rather than doing it."
      switch "--write",
             description: "Make the expected file modifications without taking any Git actions."
      switch "--commit",
             depends_on:  "--write",
             description: "When passed with `--write`, generate a new commit after writing changes "\
                          "to the cask file."
      switch "--no-audit",
             description: "Don't run `brew cask audit` before opening the PR."
      switch "--online",
             description: "Run `brew cask audit --online` before opening the PR."
      switch "--no-style",
             description: "Don't run `brew cask style --fix` before opening the PR."
      switch "--no-browse",
             description: "Print the pull request URL instead of opening in a browser."
      switch "--no-fork",
             description: "Don't try to fork the repository."
      flag   "--version=",
             description: "Specify the new <version> for the cask."
      flag   "--message=",
             description: "Append <message> to the default pull request message."
      flag   "--url=",
             description: "Specify the <URL> for the new download."
      flag   "--sha256=",
             description: "Specify the <SHA-256> checksum of the new download."
      switch "-f", "--force",
             description: "Ignore duplicate open PRs."

      conflicts "--dry-run", "--write"
      conflicts "--no-audit", "--online"
      named 1
    end
  end

  def bump_cask_pr
    args = bump_cask_pr_args.parse

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    # Use the user's browser, too.
    ENV["BROWSER"] = Homebrew::EnvConfig.browser

    cask = args.named.to_casks.first
    new_version = args.version
    new_version = "latest" if new_version == ":latest"
    new_version = Cask::DSL::Version.new(new_version)
    new_base_url = args.url
    new_hash = args.sha256

    old_version = cask.version
    old_hash = cask.sha256

    tap_full_name = cask.tap&.full_name
    origin_branch = Utils::Git.origin_branch(cask.tap.path) if cask.tap
    origin_branch ||= "origin/master"
    previous_branch = "-"

    check_open_pull_requests(cask, tap_full_name, args: args)

    odie "#{cask}: no --version= argument specified!" if new_version.empty?

    check_closed_pull_requests(cask, tap_full_name, version: new_version, args: args) unless new_version.latest?

    if new_version == old_version
      odie <<~EOS
        You need to bump this cask manually since the new version
        and old version are both #{new_version}.
      EOS
    elsif old_version.latest?
      opoo "No --url= argument specified!" unless new_base_url
    elsif new_version.latest?
      opoo "Ignoring specified --sha256= argument." if new_hash
    elsif Version.new(new_version) < Version.new(old_version)
      odie <<~EOS
        You need to bump this cask manually since changing the
        version from #{old_version} to #{new_version} would be a downgrade.
      EOS
    end

    old_contents = File.read(cask.sourcefile_path)

    replacement_pairs = []

    replacement_pairs << if old_version.latest?
      [
        "version :latest",
        "version \"#{new_version}\"",
      ]
    elsif new_version.latest?
      [
        "version \"#{old_version}\"",
        "version :latest",
      ]
    else
      [
        old_version,
        new_version,
      ]
    end

    if new_base_url.present?
      m = /^ +url "(.+?)"\n/m.match(old_contents)
      odie "Could not find old URL in cask!" if m.nil?

      old_base_url = m.captures.first

      replacement_pairs << [
        /#{Regexp.escape(old_base_url)}/,
        new_base_url,
      ]
    end

    if !new_version.latest? && (new_hash.nil? || cask.languages.present?)
      tmp_contents = Utils::Inreplace.inreplace_pairs(cask.sourcefile_path,
                                                      replacement_pairs.uniq.compact,
                                                      read_only_run: true,
                                                      silent:        true)

      tmp_cask = Cask::CaskLoader.load(tmp_contents)
      tmp_url = tmp_cask.url.to_s

      if new_hash.nil?
        resource_path = fetch_resource(cask, new_version, tmp_url)
        Utils::Tar.validate_file(resource_path)
        new_hash = resource_path.sha256
      end

      cask.languages.each do |language|
        next if language == cask.language

        tmp_cask.config.languages = [language]

        lang_cask = Cask::CaskLoader.load(tmp_contents)
        lang_url = lang_cask.url.to_s
        lang_old_hash = lang_cask.sha256

        resource_path = fetch_resource(cask, new_version, lang_url)
        Utils::Tar.validate_file(resource_path)
        lang_new_hash = resource_path.sha256

        replacement_pairs << [
          lang_old_hash,
          lang_new_hash,
        ]
      end
    end

    replacement_pairs << if old_version.latest?
      [
        "sha256 :no_check",
        "sha256 \"#{new_hash}\"",
      ]
    elsif new_version.latest?
      [
        "sha256 \"#{old_hash}\"",
        "sha256 :no_check",
      ]
    else
      [
        old_hash,
        new_hash,
      ]
    end

    Utils::Inreplace.inreplace_pairs(cask.sourcefile_path,
                                     replacement_pairs.uniq.compact,
                                     read_only_run: args.dry_run?,
                                     silent:        args.quiet?)

    run_cask_audit(cask, old_contents, args: args)
    run_cask_style(cask, old_contents, args: args)

    pr_info = {
      sourcefile_path: cask.sourcefile_path,
      old_contents:    old_contents,
      origin_branch:   origin_branch,
      branch_name:     "bump-#{cask.token}-#{new_version.tr(",:", "-")}",
      commit_message:  "Update #{cask.token} from #{old_version} to #{new_version}",
      previous_branch: previous_branch,
      tap:             cask.tap,
      tap_full_name:   tap_full_name,
      pr_message:      "Created with `brew bump-cask-pr`.",
    }
    GitHub.create_bump_pr(pr_info, args: args)
  end

  def fetch_resource(cask, new_version, url, **specs)
    resource = Resource.new
    resource.url(url, specs)
    resource.owner = Resource.new(cask.token)
    resource.version = new_version
    resource.fetch
  end

  def check_open_pull_requests(cask, tap_full_name, args:)
    GitHub.check_for_duplicate_pull_requests(cask.token, tap_full_name, state: "open", args: args)
  end

  def check_closed_pull_requests(cask, tap_full_name, version:, args:)
    # if we haven't already found open requests, try for an exact match across closed requests
    pr_title = "Update #{cask.token} from #{cask.version} to #{version}"
    GitHub.check_for_duplicate_pull_requests(pr_title, tap_full_name, state: "closed", args: args)
  end

  def run_cask_audit(cask, old_contents, args:)
    if args.dry_run?
      if args.no_audit?
        ohai "Skipping `brew cask audit`"
      elsif args.online?
        ohai "brew cask audit --online #{cask.sourcefile_path.basename}"
      else
        ohai "brew cask audit #{cask.sourcefile_path.basename}"
      end
      return
    end
    failed_audit = false
    if args.no_audit?
      ohai "Skipping `brew cask audit`"
    elsif args.online?
      system HOMEBREW_BREW_FILE, "cask", "audit", "--online", cask.sourcefile_path
      failed_audit = !$CHILD_STATUS.success?
    else
      system HOMEBREW_BREW_FILE, "cask", "audit", cask.sourcefile_path
      failed_audit = !$CHILD_STATUS.success?
    end
    return unless failed_audit

    cask.sourcefile_path.atomic_write(old_contents)
    odie "`brew cask audit` failed!"
  end

  def run_cask_style(cask, old_contents, args:)
    if args.dry_run?
      if args.no_style?
        ohai "Skipping `brew cask style --fix`"
      else
        ohai "brew cask style --fix #{cask.sourcefile_path.basename}"
      end
      return
    end
    failed_style = false
    if args.no_style?
      ohai "Skipping `brew cask style --fix`"
    else
      system HOMEBREW_BREW_FILE, "cask", "style", "--fix", cask.sourcefile_path
      failed_style = !$CHILD_STATUS.success?
    end
    return unless failed_style

    cask.sourcefile_path.atomic_write(old_contents)
    odie "`brew cask style --fix` failed!"
  end
end
