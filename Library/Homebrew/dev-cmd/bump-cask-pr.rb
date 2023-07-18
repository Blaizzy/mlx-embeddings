# typed: true
# frozen_string_literal: true

require "cask"
require "cask/download"
require "cli/parser"
require "utils/tar"

module Homebrew
  module_function

  sig { returns(CLI::Parser) }
  def bump_cask_pr_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Create a pull request to update <cask> with a new version.

        A best effort to determine the <SHA-256> will be made if the value is not
        supplied by the user.
      EOS
      switch "-n", "--dry-run",
             description: "Print what would be done rather than doing it."
      switch "--write-only",
             description: "Make the expected file modifications without taking any Git actions."
      switch "--commit",
             depends_on:  "--write-only",
             description: "When passed with `--write-only`, generate a new commit after writing changes " \
                          "to the cask file."
      switch "--no-audit",
             description: "Don't run `brew audit` before opening the PR."
      switch "--online",
             description: "Run `brew audit --online` before opening the PR."
      switch "--no-style",
             description: "Don't run `brew style --fix` before opening the PR."
      switch "--no-browse",
             description: "Print the pull request URL instead of opening in a browser."
      switch "--no-fork",
             description: "Don't try to fork the repository."
      flag   "--version=",
             description: "Specify the new <version> for the cask."
      flag   "--message=",
             description: "Prepend <message> to the default pull request message."
      flag   "--url=",
             description: "Specify the <URL> for the new download."
      flag   "--sha256=",
             description: "Specify the <SHA-256> checksum of the new download."
      flag   "--fork-org=",
             description: "Use the specified GitHub organization for forking."
      switch "-f", "--force",
             description: "Ignore duplicate open PRs."

      conflicts "--dry-run", "--write"
      conflicts "--no-audit", "--online"

      named_args :cask, number: 1, without_api: true
    end
  end

  def bump_cask_pr
    args = bump_cask_pr_args.parse

    # This will be run by `brew audit` or `brew style` later so run it first to
    # not start spamming during normal output.
    Homebrew.install_bundler_gems! if !args.no_audit? || !args.no_style?

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = PATH.new(ORIGINAL_PATHS).to_s

    # Use the user's browser, too.
    ENV["BROWSER"] = Homebrew::EnvConfig.browser

    cask = args.named.to_casks.first

    odie "This cask is not in a tap!" if cask.tap.blank?
    odie "This cask's tap is not a Git repository!" unless cask.tap.git?

    new_version = unless (new_version = args.version).nil?
      raise UsageError, "`--version` must not be empty." if new_version.blank?

      new_version = :latest if ["latest", ":latest"].include?(new_version)
      Cask::DSL::Version.new(new_version)
    end

    new_hash = unless (new_hash = args.sha256).nil?
      raise UsageError, "`--sha256` must not be empty." if new_hash.blank?

      ["no_check", ":no_check"].include?(new_hash) ? :no_check : new_hash
    end

    new_base_url = unless (new_base_url = args.url).nil?
      raise UsageError, "`--url` must not be empty." if new_base_url.blank?

      begin
        URI(new_base_url)
      rescue URI::InvalidURIError
        raise UsageError, "`--url` is not valid."
      end
    end

    if new_version.nil? && new_base_url.nil? && new_hash.nil?
      raise UsageError, "No `--version`, `--url` or `--sha256` argument specified!"
    end

    old_version = cask.version
    old_hash = cask.sha256

    check_pull_requests(cask, state: "open", args: args)
    # if we haven't already found open requests, try for an exact match across closed requests
    check_pull_requests(cask, state: "closed", args: args, version: new_version) if new_version.present?

    old_contents = File.read(cask.sourcefile_path)

    replacement_pairs = []
    branch_name = "bump-#{cask.token}"
    commit_message = nil

    if new_version
      branch_name += "-#{new_version.tr(",:", "-")}"
      commit_message_version = if new_version.before_comma == old_version.before_comma
        new_version
      else
        new_version.before_comma
      end
      commit_message ||= "#{cask.token} #{commit_message_version}"

      old_version_regex = old_version.latest? ? ":latest" : "[\"']#{Regexp.escape(old_version.to_s)}[\"']"

      replacement_pairs << [
        /version\s+#{old_version_regex}/m,
        "version #{new_version.latest? ? ":latest" : "\"#{new_version}\""}",
      ]
      if new_version.latest? || new_hash == :no_check
        opoo "Ignoring specified `--sha256=` argument." if new_hash.is_a?(String)
        replacement_pairs << [/"#{old_hash}"/, ":no_check"] if old_hash != :no_check
      elsif old_hash == :no_check && new_hash != :no_check
        replacement_pairs << [":no_check", "\"#{new_hash}\""] if new_hash.is_a?(String)
      elsif old_hash != :no_check
        if new_hash.nil? || cask.languages.present?
          if new_hash && cask.languages.present?
            opoo "Multiple checksum replacements required; ignoring specified `--sha256` argument."
          end
          tmp_contents = Utils::Inreplace.inreplace_pairs(cask.sourcefile_path,
                                                          replacement_pairs.uniq.compact,
                                                          read_only_run: true,
                                                          silent:        true)

          tmp_cask = Cask::CaskLoader.load(tmp_contents)
          tmp_config = tmp_cask.config

          OnSystem::ARCH_OPTIONS.each do |arch|
            SimulateSystem.with arch: arch do
              languages = cask.languages
              languages = [nil] if languages.empty?
              languages.each do |language|
                new_hash_config = if language.blank?
                  tmp_config
                else
                  tmp_config.merge(Cask::Config.new(explicit: { languages: [language] }))
                end

                new_hash_cask = Cask::CaskLoader.load(tmp_contents)
                new_hash_cask.config = new_hash_config
                old_hash = new_hash_cask.sha256.to_s

                cask_download = Cask::Download.new(new_hash_cask, quarantine: true)
                download = cask_download.fetch(verify_download_integrity: false)
                Utils::Tar.validate_file(download)

                replacement_pairs << [new_hash_cask.sha256.to_s, download.sha256]
              end
            end
          end
        elsif new_hash
          opoo "Cask contains multiple hashes; only updating hash for current arch." if cask.on_system_blocks_exist?
          replacement_pairs << [old_hash.to_s, new_hash]
        end
      end
    end

    if new_base_url
      commit_message ||= "#{cask.token}: update URL"

      m = /^ +url "(.+?)"\n/m.match(old_contents)
      odie "Could not find old URL in cask!" if m.nil?

      old_base_url = m.captures.fetch(0)

      replacement_pairs << [
        /#{Regexp.escape(old_base_url)}/,
        new_base_url.to_s,
      ]
    end

    commit_message ||= "#{cask.token}: update checksum" if new_hash

    Utils::Inreplace.inreplace_pairs(cask.sourcefile_path,
                                     replacement_pairs.uniq.compact,
                                     read_only_run: args.dry_run?,
                                     silent:        args.quiet?)

    run_cask_audit(cask, old_contents, args: args)
    run_cask_style(cask, old_contents, args: args)

    pr_info = {
      sourcefile_path: cask.sourcefile_path,
      old_contents:    old_contents,
      branch_name:     branch_name,
      commit_message:  commit_message,
      tap:             cask.tap,
      pr_message:      "Created with `brew bump-cask-pr`.",
    }
    GitHub.create_bump_pr(pr_info, args: args)
  end

  def check_pull_requests(cask, state:, args:, version: nil)
    tap_remote_repo = cask.tap.full_name || cask.tap.remote_repo
    GitHub.check_for_duplicate_pull_requests(cask.token, tap_remote_repo,
                                             state:   state,
                                             version: version,
                                             file:    cask.sourcefile_path.relative_path_from(cask.tap.path).to_s,
                                             args:    args)
  end

  def run_cask_audit(cask, old_contents, args:)
    if args.dry_run?
      if args.no_audit?
        ohai "Skipping `brew audit`"
      elsif args.online?
        ohai "brew audit --cask --online #{cask.full_name}"
      else
        ohai "brew audit --cask #{cask.full_name}"
      end
      return
    end
    failed_audit = false
    if args.no_audit?
      ohai "Skipping `brew audit`"
    elsif args.online?
      system HOMEBREW_BREW_FILE, "audit", "--cask", "--online", cask.full_name
      failed_audit = !$CHILD_STATUS.success?
    else
      system HOMEBREW_BREW_FILE, "audit", "--cask", cask.full_name
      failed_audit = !$CHILD_STATUS.success?
    end
    return unless failed_audit

    cask.sourcefile_path.atomic_write(old_contents)
    odie "`brew audit` failed!"
  end

  def run_cask_style(cask, old_contents, args:)
    if args.dry_run?
      if args.no_style?
        ohai "Skipping `brew style --fix`"
      else
        ohai "brew style --fix #{cask.sourcefile_path.basename}"
      end
      return
    end
    failed_style = false
    if args.no_style?
      ohai "Skipping `brew style --fix`"
    else
      system HOMEBREW_BREW_FILE, "style", "--fix", cask.sourcefile_path
      failed_style = !$CHILD_STATUS.success?
    end
    return unless failed_style

    cask.sourcefile_path.atomic_write(old_contents)
    odie "`brew style --fix` failed!"
  end
end
