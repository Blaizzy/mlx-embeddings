# typed: true
# frozen_string_literal: true

require "cli/parser"
require "csv"

module Homebrew
  extend T::Sig

  module_function

  PRIMARY_REPOS = %w[brew core cask].freeze
  SUPPORTED_REPOS = [
    PRIMARY_REPOS,
    OFFICIAL_CMD_TAPS.keys.map { |t| t.delete_prefix("homebrew/") },
    OFFICIAL_CASK_TAPS.reject { |t| t == "cask" },
  ].flatten.freeze

  sig { returns(CLI::Parser) }
  def contributions_args
    Homebrew::CLI::Parser.new do
      usage_banner "`contributions` [--user=<email|username>] [<--repositories>`=`] [<--csv>]"
      description <<~EOS
        Contributions to Homebrew repos.
      EOS

      comma_array "--repositories",
                  description: "Specify a comma-separated (no spaces) list of repositories to search. " \
                               "Supported repositories: #{SUPPORTED_REPOS.map { |t| "`#{t}`" }.to_sentence}. " \
                               "Omitting this flag, or specifying `--repositories=all`, searches all repositories. " \
                               "Use `--repositories=primary` to search only the main repositories: brew,core,cask."
      flag "--from=",
           description: "Date (ISO-8601 format) to start searching contributions."

      flag "--to=",
           description: "Date (ISO-8601 format) to stop searching contributions."

      flag "--user=",
           description: "A GitHub username or email address of a specific person to find contribution data for."

      switch "--csv",
             description: "Print a CSV of a user's contributions across repositories over the time period."
    end
  end

  sig { void }
  def contributions
    args = contributions_args.parse

    results = {}

    all_repos = args.repositories.nil? || args.repositories.include?("all")
    repos = if all_repos
      SUPPORTED_REPOS
    elsif args.repositories.include?("primary")
      PRIMARY_REPOS
    else
      args.repositories
    end

    if args.user
      user = args.user
      results[user] = scan_repositories(repos, user, args)
      puts "#{user} contributed #{total(results[user])} times #{time_period(args)}."
      puts generate_csv(T.must(user), results[user]) if args.csv?
      return
    end

    odie "CSVs not yet supported for the full list of maintainers at once." if args.csv?

    maintainers = GitHub.members_by_team("Homebrew", "maintainers")
    maintainers.each do |username, _|
      # TODO: Using the GitHub username to scan the `git log` undercounts some
      # contributions as people might not always have configured their Git
      # committer details to match the ones on GitHub.
      # TODO: Switch to using the GitHub APIs instead of `git log` if
      # they ever support trailers.
      results[username] = scan_repositories(repos, username, args)
      puts "#{username} contributed #{total(results[username])} times #{time_period(args)}."
    end
  end

  sig { params(repo: String).returns(Pathname) }
  def find_repo_path_for_repo(repo)
    return HOMEBREW_REPOSITORY if repo == "brew"

    Tap.fetch("homebrew", repo).path
  end

  sig { params(args: Homebrew::CLI::Args).returns(String) }
  def time_period(args)
    if args.from && args.to
      "between #{args.from} and #{args.to}"
    elsif args.from
      "after #{args.from}"
    elsif args.to
      "before #{args.to}"
    else
      "in all time"
    end
  end

  sig { params(user: String, results: Hash).returns(String) }
  def generate_csv(user, results)
    CSV.generate do |csv|
      csv << %w[user repo commits coauthorships signoffs total]
      results.each do |repo, counts|
        csv << [
          user,
          repo,
          counts[:commits],
          counts[:coauthorships],
          counts[:signoffs],
          counts.values.sum,
        ]
      end
      csv << [user, "*", "*", "*", "*", total(results)]
    end
  end

  def scan_repositories(repos, person, args)
    data = {}

    repos.each do |repo|
      if SUPPORTED_REPOS.exclude?(repo)
        return ofail "Unsupported repository: #{repo}. Try one of #{SUPPORTED_REPOS.join(", ")}."
      end

      repo_path = find_repo_path_for_repo(repo)
      tap = Tap.fetch("homebrew", repo)
      unless repo_path.exist?
        opoo "Repository #{repo} not yet tapped! Tapping it now..."
        tap.install
      end

      repo_full_name = if repo == "brew"
        "homebrew/brew"
      else
        tap.full_name
      end

      puts "Determining contributions for #{person} on #{repo_full_name}..." if args.verbose?

      data[repo] = {
        commits:       GitHub.repo_commit_count_for_user(repo_full_name, person, args),
        coauthorships: git_log_trailers_cmd(T.must(repo_path), person, "Co-authored-by", args),
        signoffs:      git_log_trailers_cmd(T.must(repo_path), person, "Signed-off-by", args),
      }
    end

    data
  end

  sig { params(results: Hash).returns(Integer) }
  def total(results)
    results
      .values # [{:commits=>1, :coauthorships=>0, :signoffs=>3}, {:commits=>500, :coauthorships=>2, :signoffs=>450}]
      .map(&:values) # [[1, 0, 3], [500, 2, 450]]
      .sum(&:sum) # 956
  end

  sig { params(repo_path: Pathname, person: String, trailer: String, args: Homebrew::CLI::Args).returns(Integer) }
  def git_log_trailers_cmd(repo_path, person, trailer, args)
    cmd = ["git", "-C", repo_path, "log", "--oneline"]
    cmd << "--format='%(trailers:key=#{trailer}:)'"
    cmd << "--before=#{args.to}" if args.to
    cmd << "--after=#{args.from}" if args.from

    Utils.safe_popen_read(*cmd).lines.count { |l| l.include?(person) }
  end
end
