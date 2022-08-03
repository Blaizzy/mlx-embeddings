# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  SUPPORTED_REPOS = [
    %w[brew core cask],
    OFFICIAL_CMD_TAPS.keys.map { |t| t.delete_prefix("homebrew/") },
    OFFICIAL_CASK_TAPS.reject { |t| t == "cask" },
  ].flatten.freeze

  sig { returns(CLI::Parser) }
  def contributions_args
    Homebrew::CLI::Parser.new do
      usage_banner "`contributions` <email|name> [<--repositories>`=`]"
      description <<~EOS
        Contributions to Homebrew repos for a user.

        The first argument is a name (e.g. "BrewTestBot") or an email address (e.g. "brewtestbot@brew.sh").
      EOS

      comma_array "--repositories",
                  description: "Specify a comma-separated (no spaces) list of repositories to search. " \
                               "Supported repositories: #{SUPPORTED_REPOS.map { |t| "`#{t}`" }.to_sentence}." \
                               "Omitting this flag, or specifying `--repositories=all`, will search all repositories."
      flag "--from=",
           description: "Date (ISO-8601 format) to start searching contributions."

      flag "--to=",
           description: "Date (ISO-8601 format) to stop searching contributions."

      named_args number: 1
    end
  end

  sig { void }
  def contributions
    args = contributions_args.parse

    commits = 0
    coauthorships = 0

    all_repos = args.repositories.nil? || args.repositories.include?("all")
    repos = all_repos ? SUPPORTED_REPOS : args.repositories

    repos.each do |repo|
      if SUPPORTED_REPOS.exclude?(repo)
        return ofail "Unsupported repository: #{repo}. Try one of #{SUPPORTED_REPOS.join(", ")}."
      end

      repo_path = find_repo_path_for_repo(repo)
      unless repo_path.exist?

        opoo "Repository #{repo} not yet tapped! Tapping it now..."
        Tap.fetch("homebrew", repo).install
      end

      commits += git_log_author_cmd(T.must(repo_path), args)
      coauthorships += git_log_coauthor_cmd(T.must(repo_path), args)
    end

    sentence = "#{args.named.first} directly authored #{commits} commits " \
               "and co-authored #{coauthorships} commits " \
               "across #{all_repos ? "all Homebrew repos" : repos.to_sentence}"
    sentence += if args.from && args.to
      " between #{args.from} and #{args.to}"
    elsif args.from
      " after #{args.from}"
    elsif args.to
      " before #{args.to}"
    else
      " in all time"
    end
    sentence += "."

    puts sentence
  end

  sig { params(repo: String).returns(Pathname) }
  def find_repo_path_for_repo(repo)
    return HOMEBREW_REPOSITORY if repo == "brew"

    Tap.fetch("homebrew", repo).path
  end

  sig { params(repo_path: Pathname, args: Homebrew::CLI::Args).returns(Integer) }
  def git_log_author_cmd(repo_path, args)
    cmd = ["git", "-C", repo_path, "log", "--oneline", "--author=#{args.named.first}"]
    cmd << "--before=#{args.to}" if args.to
    cmd << "--after=#{args.from}" if args.from

    Utils.safe_popen_read(*cmd).lines.count
  end

  sig { params(repo_path: Pathname, args: Homebrew::CLI::Args).returns(Integer) }
  def git_log_coauthor_cmd(repo_path, args)
    cmd = ["git", "-C", repo_path, "log", "--oneline"]
    cmd << "--format='%(trailers:key=Co-authored-by:)'"
    cmd << "--before=#{args.to}" if args.to
    cmd << "--after=#{args.from}" if args.from

    Utils.safe_popen_read(*cmd).lines.count { |l| l.include?(args.named.first) }
  end
end
