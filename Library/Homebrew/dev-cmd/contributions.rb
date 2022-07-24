# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  SUPPORTED_REPOS = %w[brew core cask bundle].freeze

  sig { returns(CLI::Parser) }
  def contributions_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `contributions`

        Contributions to Homebrew repos for a user.
      EOS

      flag "--email=",
           description: "A user's email address that they commit with."

      flag "--from=",
           description: "Date (ISO-8601 format) to start searching contributions."

      flag "--to=",
           description: "Date (ISO-8601 format) to stop searching contributions."

      comma_array "--repos=",
                  description: "The Homebrew repositories to search for contributions in. " \
                               "Comma separated. Supported repos: #{SUPPORTED_REPOS.join(", ")}."

      named_args :none
    end
  end

  sig { returns(NilClass) }
  def contributions
    args = contributions_args.parse

    return ofail "`--repos` and `--email` are required." if !args[:repos] || !args[:email]

    commits = 0
    coauthorships = 0

    args[:repos].each do |repo|
      repo_location = find_repo_path_for_repo(repo)
      unless repo_location
        return ofail "Couldn't find location for #{repo}. Do you have it tapped, or is there a typo? " \
                     "We only support #{SUPPORTED_REPOS.join(", ")} repos so far."
      end

      commits += git_log_cmd("author", repo_location, args)
      coauthorships += git_log_cmd("coauthorships", repo_location, args)
    end

    sentence = "Person #{args[:email]} directly authored #{commits} commits"
    sentence += " and co-authored #{coauthorships} commits"
    sentence += " to #{args[:repos].join(", ")}"
    sentence += if args[:from] && args[:to]
      " between #{args[:from]} and #{args[:to]}"
    elsif args[:from]
      " after #{args[:from]}"
    elsif args[:to]
      " before #{args[:to]}"
    else
      " in all time"
    end
    sentence += "."

    puts sentence
  end

  sig { params(repo: String).returns(T.nilable(String)) }
  def find_repo_path_for_repo(repo)
    case repo
    when "brew"
      HOMEBREW_REPOSITORY
    when "core"
      "#{HOMEBREW_REPOSITORY}/Library/Taps/homebrew/homebrew-core"
    when "cask"
      "#{HOMEBREW_REPOSITORY}/Library/Taps/homebrew/homebrew-cask"
    when "bundle"
      "#{HOMEBREW_REPOSITORY}/Library/Taps/homebrew/homebrew-bundle"
    end
  end

  sig { params(kind: String, repo_location: String, args: Homebrew::CLI::Args).returns(Integer) }
  def git_log_cmd(kind, repo_location, args)
    cmd = "git -C #{repo_location} log --oneline"
    cmd += " --author=#{args[:email]}" if kind == "author"
    cmd += " --format='%(trailers:key=Co-authored-by:)'" if kind == "coauthorships"
    cmd += " --before=#{args[:to]}" if args[:to]
    cmd += " --after=#{args[:from]}" if args[:from]
    cmd += " | grep #{args[:email]}" if kind == "coauthorships"

    `#{cmd} | wc -l`.strip.to_i
  end
end
