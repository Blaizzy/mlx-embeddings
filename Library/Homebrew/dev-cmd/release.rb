# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def release_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Create a new draft Homebrew/brew release with the appropriate version number and release notes.

        By default, `brew release` will bump the patch version number. Pass
        `--major` or `--minor` to bump the major or minor version numbers, respectively.
        The command will fail if the previous major or minor release was made less than
        one month ago.

        Requires write access to the Homebrew/brew repository.
      EOS
      switch "--major",
             description: "Create a major release."
      switch "--minor",
             description: "Create a minor release."
      conflicts "--major", "--minor"

      named_args :none
    end
  end

  def release
    args = release_args.parse

    safe_system "git", "-C", HOMEBREW_REPOSITORY, "fetch", "origin" if Homebrew::EnvConfig.no_auto_update?

    begin
      latest_release = GitHub.get_latest_release "Homebrew", "brew"
    rescue GitHub::HTTPNotFoundError
      odie "No existing releases found!"
    end
    latest_version = Version.new latest_release["tag_name"]

    if args.major? || args.minor?
      one_month_ago = Date.today << 1
      latest_major_minor_release = begin
        GitHub.get_release "Homebrew", "brew", "#{latest_version.major_minor}.0"
      rescue GitHub::HTTPNotFoundError
        nil
      end

      if latest_major_minor_release.blank?
        opoo "Unable to determine the release date of the latest major/minor release."
      elsif Date.parse(latest_major_minor_release["published_at"]) > one_month_ago
        odie "The latest major/minor release was less than one month ago."
      end
    end

    new_version = if args.major?
      Version.new [latest_version.major.to_i + 1, 0, 0].join(".")
    elsif args.minor?
      Version.new [latest_version.major, latest_version.minor.to_i + 1, 0].join(".")
    else
      Version.new [latest_version.major, latest_version.minor, latest_version.patch.to_i + 1].join(".")
    end.to_s

    ohai "Creating draft release for version #{new_version}"
    release_notes = if args.major? || args.minor?
      ["Release notes for this release can be found on the [Homebrew blog](https://brew.sh/blog/#{new_version})."]
    else
      []
    end
    release_notes += Utils.popen_read(
      "git", "-C", HOMEBREW_REPOSITORY, "log", "--pretty=format:'%s >> - %b%n'", "#{latest_version}..origin/HEAD"
    ).lines.grep(/Merge pull request/).map! do |s|
      pr = s.gsub(%r{.*Merge pull request #(\d+) from ([^/]+)/[^>]*(>>)*},
                  "https://github.com/Homebrew/brew/pull/\\1 (@\\2)")
      /(.*\d)+ \(@(.+)\) - (.*)/ =~ pr
      "- [#{Regexp.last_match(3)}](#{Regexp.last_match(1)}) (@#{Regexp.last_match(2)})"
    end

    begin
      release = GitHub.create_or_update_release "Homebrew", "brew", new_version,
                                                body: release_notes.join("\n"), draft: true
    rescue *GitHub::API_ERRORS => e
      odie "Unable to create release: #{e.message}!"
    end

    puts release["html_url"]
    exec_browser release["html_url"]
  end
end
