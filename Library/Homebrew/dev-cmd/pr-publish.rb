# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  module_function

  def pr_publish_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pr-publish` <pull_request>

        Publishes bottles for a pull request with GitHub Actions.
        Requires write access to the repository.
      EOS
      switch :verbose
    end
  end

  def pr_publish
    pr_publish_args.parse

    odie "You need to specify at least one pull request number!" if Homebrew.args.named.empty?

    args.named.each do |arg|
      arg = "#{CoreTap.instance.default_remote}/pull/#{arg}" if arg.to_i.positive?
      url_match = arg.match HOMEBREW_PULL_OR_COMMIT_URL_REGEX
      _, user, repo, issue = *url_match
      tap = Tap.fetch(user, repo) if repo.match?(HOMEBREW_OFFICIAL_REPO_PREFIXES_REGEX)
      odie "Not a GitHub pull request: #{arg}" unless issue
      ohai "Dispatching #{tap} pull request ##{issue}"
      GitHub.dispatch_event(user, repo, "Publish ##{issue}", pull_request: issue)
    end
  end
end
