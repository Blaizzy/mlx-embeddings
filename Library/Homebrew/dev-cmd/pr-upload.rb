# typed: false
# frozen_string_literal: true

require "cli/parser"
require "archive"
require "github_packages"
require "github_releases"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def pr_upload_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Apply the bottle commit and publish bottles to a host.
      EOS
      switch "--keep-old",
             description: "If the formula specifies a rebuild version, " \
                          "attempt to preserve its value in the generated DSL."
      switch "-n", "--dry-run",
             description: "Print what would be done rather than doing it."
      switch "--no-commit",
             description: "Do not generate a new commit before uploading."
      switch "--warn-on-upload-failure",
             description: "Warn instead of raising an error if the bottle upload fails. "\
                          "Useful for repairing bottle uploads that previously failed."
      flag   "--committer=",
             description: "Specify a committer name and email in `git`'s standard author format."
      flag   "--archive-item=",
             description: "Upload to the specified Internet Archive item (default: `homebrew`)."
      flag   "--github-org=",
             description: "Upload to the specified GitHub organisation's GitHub Packages (default: `homebrew`)."
      flag   "--root-url=",
             description: "Use the specified <URL> as the root of the bottle's URL instead of Homebrew's default."
      flag   "--root-url-using=",
             description: "Use the specified download strategy class for downloading the bottle's URL instead of "\
                          "Homebrew's default."

      named_args :none
    end
  end

  def check_bottled_formulae!(bottles_hash)
    bottles_hash.each do |name, bottle_hash|
      formula_path = HOMEBREW_REPOSITORY/bottle_hash["formula"]["path"]
      formula_version = Formulary.factory(formula_path).pkg_version
      bottle_version = PkgVersion.parse bottle_hash["formula"]["pkg_version"]
      next if formula_version == bottle_version

      odie "Bottles are for #{name} #{bottle_version} but formula is version #{formula_version}!"
    end
  end

  def internet_archive?(bottles_hash)
    @internet_archive ||= bottles_hash.values.all? do |bottle_hash|
      bottle_hash["bottle"]["root_url"].start_with? "#{Archive::URL_PREFIX}/"
    end
  end

  def github_releases?(bottles_hash)
    @github_releases ||= bottles_hash.values.all? do |bottle_hash|
      root_url = bottle_hash["bottle"]["root_url"]
      url_match = root_url.match GitHubReleases::URL_REGEX
      _, _, _, tag = *url_match

      tag
    end
  end

  def github_packages?(bottles_hash)
    @github_packages ||= bottles_hash.values.all? do |bottle_hash|
      bottle_hash["bottle"]["root_url"].match? GitHubPackages::URL_REGEX
    end
  end

  def bottles_hash_from_json_files(json_files, args)
    puts "Reading JSON files: #{json_files.join(", ")}" if args.verbose?

    bottles_hash = json_files.reduce({}) do |hash, json_file|
      hash.deep_merge(JSON.parse(File.read(json_file)))
    end

    if args.root_url
      bottles_hash.each_value do |bottle_hash|
        bottle_hash["bottle"]["root_url"] = args.root_url
      end
    end

    bottles_hash
  end

  def pr_upload
    args = pr_upload_args.parse

    json_files = Dir["*.bottle.json"]
    odie "No bottle JSON files found in the current working directory" if json_files.blank?
    bottles_hash = bottles_hash_from_json_files(json_files, args)

    bottle_args = ["bottle", "--merge", "--write"]
    bottle_args << "--verbose" if args.verbose?
    bottle_args << "--debug" if args.debug?
    bottle_args << "--keep-old" if args.keep_old?
    bottle_args << "--root-url=#{args.root_url}" if args.root_url
    bottle_args << "--committer=#{args.committer}" if args.committer
    bottle_args << "--no-commit" if args.no_commit?
    bottle_args << "--root-url-using=#{args.root_url_using}" if args.root_url_using
    bottle_args += json_files

    if args.dry_run?
      dry_run_service = if github_packages?(bottles_hash)
        # GitHub Packages has its own --dry-run handling.
        nil
      elsif internet_archive?(bottles_hash)
        "Internet Archive"
      elsif github_releases?(bottles_hash)
        "GitHub Releases"
      else
        odie "Service specified by root_url is not recognized"
      end

      if dry_run_service
        puts <<~EOS
          brew #{bottle_args.join " "}
          Upload bottles described by these JSON files to #{dry_run_service}:
            #{json_files.join("\n  ")}
        EOS
        return
      end
    end

    check_bottled_formulae!(bottles_hash)

    # This will be run by `brew bottle` and `brew audit` later so run it first
    # to not start spamming during normal output.
    Homebrew.install_bundler_gems!

    safe_system HOMEBREW_BREW_FILE, *bottle_args

    json_files = Dir["*.bottle.json"]
    if json_files.blank?
      puts "No bottle JSON files after merge, no upload needed!"
      return
    end

    # Reload the JSON files (in case `brew bottle --merge` generated
    # `all: $SHA256` bottles)
    bottles_hash = bottles_hash_from_json_files(json_files, args)

    # Check the bottle commits did not break `brew audit`
    unless args.no_commit?
      audit_args = ["audit", "--skip-style"]
      audit_args << "--verbose" if args.verbose?
      audit_args << "--debug" if args.debug?
      audit_args += bottles_hash.keys
      safe_system HOMEBREW_BREW_FILE, *audit_args
    end

    if internet_archive?(bottles_hash)
      archive_item = args.archive_item || "homebrew"
      archive = Archive.new(item: archive_item)
      archive.upload_bottles(bottles_hash,
                             warn_on_error: args.warn_on_upload_failure?)
    elsif github_releases?(bottles_hash)
      github_releases = GitHubReleases.new
      github_releases.upload_bottles(bottles_hash)
    elsif github_packages?(bottles_hash)
      github_org = args.github_org || "homebrew"
      github_packages = GitHubPackages.new(org: github_org)
      github_packages.upload_bottles(bottles_hash,
                                     keep_old:      args.keep_old?,
                                     dry_run:       args.dry_run?,
                                     warn_on_error: args.warn_on_upload_failure?)
    else
      odie "Service specified by root_url is not recognized"
    end
  end
end
