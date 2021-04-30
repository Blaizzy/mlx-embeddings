# typed: false
# frozen_string_literal: true

require "cask"
require "cli/parser"
require "utils/tar"

module Homebrew
  extend T::Sig

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
      switch "--write",
             description: "Make the expected file modifications without taking any Git actions."
      switch "--commit",
             depends_on:  "--write",
             description: "When passed with `--write`, generate a new commit after writing changes "\
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
             description: "Append <message> to the default pull request message."
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

      named_args :cask, number: 1
    end
  end

  def bump_cask_pr
    args = bump_cask_pr_args.parse

    # This will be run by `brew style` later so run it first to not start
    # spamming during normal output.
    Homebrew.install_bundler_gems!

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    # Use the user's browser, too.
    ENV["BROWSER"] = Homebrew::EnvConfig.browser

    cask = args.named.to_casks.first

    odie "This cask is not in a tap!" if cask.tap.blank?
    odie "This cask's tap is not a Git repository!" unless cask.tap.git?

    new_version = args.version
    new_version = :latest if ["latest", ":latest"].include?(new_version)
    new_version = Cask::DSL::Version.new(new_version) if new_version.present?
    new_base_url = args.url
    new_hash = args.sha256
    new_hash = :no_check if ["no_check", ":no_check"].include? new_hash

    if new_version.nil? && new_base_url.nil? && new_hash.nil?
      raise UsageError, "No --version=/--url=/--sha256= argument specified!"
    end

    old_version = cask.version
    old_hash = cask.sha256

    check_open_pull_requests(cask, args: args)

    old_contents = File.read(cask.sourcefile_path)

    replacement_pairs = []

    if new_version.present?
      old_version_regex = old_version.latest? ? ":latest" : "[\"']#{Regexp.escape(old_version.to_s)}[\"']"

      replacement_pairs << [
        /version\s+#{old_version_regex}/m,
        "version #{new_version.latest? ? ":latest" : "\"#{new_version}\""}",
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

    if new_version.present?
      if new_version.latest?
        opoo "Ignoring specified `--sha256=` argument." if new_hash.present?
        new_hash = :no_check
      elsif new_hash.nil? || cask.languages.present?
        tmp_contents = Utils::Inreplace.inreplace_pairs(cask.sourcefile_path,
                                                        replacement_pairs.uniq.compact,
                                                        read_only_run: true,
                                                        silent:        true)

        tmp_cask = Cask::CaskLoader.load(tmp_contents)
        tmp_config = cask.config
        tmp_url = tmp_cask.url.to_s

        if old_hash != :no_check
          new_hash = fetch_resource(cask, new_version, tmp_url) if new_hash.nil?

          if tmp_contents.include?("Hardware::CPU.intel?")
            other_intel = !Hardware::CPU.intel?
            other_contents = tmp_contents.gsub("Hardware::CPU.intel?", other_intel.to_s)
            replacement_pairs << fetch_cask(other_contents, new_version)
          end
        end

        cask.languages.each do |language|
          next if language == cask.language

          lang_config = tmp_config.merge(Cask::Config.new(explicit: { languages: [language] }))
          replacement_pairs << fetch_cask(tmp_contents, new_version, config: lang_config)
        end
      end
    end

    if new_hash.present?
      hash_regex = old_hash == :no_check ? ":no_check" : "[\"']#{Regexp.escape(old_hash.to_s)}[\"']"

      replacement_pairs << [
        /sha256\s+#{hash_regex}/m,
        "sha256 #{new_hash == :no_check ? ":no_check" : "\"#{new_hash}\""}",
      ]
    end

    Utils::Inreplace.inreplace_pairs(cask.sourcefile_path,
                                     replacement_pairs.uniq.compact,
                                     read_only_run: args.dry_run?,
                                     silent:        args.quiet?)

    run_cask_audit(cask, old_contents, args: args)
    run_cask_style(cask, old_contents, args: args)

    branch_name = "bump-#{cask.token}"
    commit_message = "Update #{cask.token}"
    if new_version.present?
      branch_name += "-#{new_version.tr(",:", "-")}"
      commit_message += " from #{old_version} to #{new_version}"
    end
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

  def fetch_resource(cask, version, url, **specs)
    resource = Resource.new
    resource.url(url, specs)
    resource.owner = Resource.new(cask.token)
    resource.version = version

    resource_path = resource.fetch
    Utils::Tar.validate_file(resource_path)
    resource_path.sha256
  end

  def fetch_cask(contents, version, config: nil)
    cask = Cask::CaskLoader.load(contents)
    cask.config = config if config.present?
    url = cask.url.to_s
    old_hash = cask.sha256.to_s
    new_hash = fetch_resource(cask, version, url)
    [old_hash, new_hash]
  end

  def check_open_pull_requests(cask, args:)
    tap_remote_repo = cask.tap.remote_repo || cask.tap.full_name
    GitHub.check_for_duplicate_pull_requests(cask.token, tap_remote_repo,
                                             state: "open",
                                             file:  cask.sourcefile_path.relative_path_from(cask.tap.path).to_s,
                                             args:  args)
  end

  def run_cask_audit(cask, old_contents, args:)
    if args.dry_run?
      if args.no_audit?
        ohai "Skipping `brew audit`"
      elsif args.online?
        ohai "brew audit --cask --online #{cask.sourcefile_path.basename}"
      else
        ohai "brew audit --cask #{cask.sourcefile_path.basename}"
      end
      return
    end
    failed_audit = false
    if args.no_audit?
      ohai "Skipping `brew audit`"
    elsif args.online?
      system HOMEBREW_BREW_FILE, "audit", "--cask", "--online", cask.sourcefile_path
      failed_audit = !$CHILD_STATUS.success?
    else
      system HOMEBREW_BREW_FILE, "audit", "--cask", cask.sourcefile_path
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
