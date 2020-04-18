# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  module_function

  def pr_publish_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pr-publish` [<options>] <pull_request> [<pull_request> ...]

        Publish bottles for a pull request with GitHub Actions.
        Requires write access to the `homebrew/core` repository.
      EOS
      switch :verbose
      min_named 1
    end
  end

  def pr_publish
    pr_publish_args.parse

    ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"] = "1" unless OS.mac?

    args.named.uniq.each do |arg|
      arg = "#{CoreTap.instance.default_remote}/pull/#{arg}" if arg.to_i.positive?
      url_match = arg.match HOMEBREW_PULL_OR_COMMIT_URL_REGEX
      _, user, repo, issue = *url_match
      odie "Not a GitHub pull request: #{arg}" unless issue
      tap = Tap.fetch(user, repo) if repo.match?(HOMEBREW_OFFICIAL_REPO_PREFIXES_REGEX)
      ohai "Dispatching #{tap} pull request ##{issue}"
      GitHub.dispatch_event(user, repo, "Publish ##{issue}", pull_request: issue)
    end
  end
end
