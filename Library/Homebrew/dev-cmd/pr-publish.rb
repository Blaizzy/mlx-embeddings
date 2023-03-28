# typed: true
# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def pr_publish_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Publish bottles for a pull request with GitHub Actions.
        Requires write access to the repository.
      EOS
      switch "--commit-bottles-to-pr-branch",
             description: "Push bottle commits to the pull request branch."
      switch "--autosquash",
             description: "If supported on the target tap, automatically reformat and reword commits " \
                          "to our preferred format."
      switch "--no-autosquash",
             description: "Skip automatically reformatting and rewording commits in the pull request " \
                          "to the preferred format, even if supported on the target tap.",
             replacement: "`--autosquash` to opt in"
      switch "--large-runner",
             description: "Run the upload job on a large runner."
      flag   "--branch=",
             description: "Branch to use the workflow from (default: `master`)."
      flag   "--message=",
             depends_on:  "--autosquash",
             description: "Message to include when autosquashing revision bumps, deletions, and rebuilds."
      flag   "--tap=",
             description: "Target tap repository (default: `homebrew/core`)."
      flag   "--workflow=",
             description: "Target workflow filename (default: `publish-commit-bottles.yml`)."

      conflicts "--clean", "--autosquash"

      named_args :pull_request, min: 1
    end
  end

  def pr_publish
    args = pr_publish_args.parse

    odeprecated "`brew pr-publish --no-autosquash`" if args.no_autosquash?

    tap = Tap.fetch(args.tap || CoreTap.instance.name)
    workflow = args.workflow || "publish-commit-bottles.yml"
    ref = args.branch || "master"

    inputs = {
      commit_bottles_to_pr_branch: args.commit_bottles_to_pr_branch?,
      autosquash:                  args.autosquash?,
      large_runner:                args.large_runner?,
    }
    inputs[:message] = args.message if args.message.presence

    args.named.uniq.each do |arg|
      arg = "#{tap.default_remote}/pull/#{arg}" if arg.to_i.positive?
      url_match = arg.match HOMEBREW_PULL_OR_COMMIT_URL_REGEX
      _, user, repo, issue = *url_match
      odie "Not a GitHub pull request: #{arg}" unless issue

      inputs[:pull_request] = issue

      if args.tap.present? && !T.must("#{user}/#{repo}".casecmp(tap.full_name)).zero?
        odie "Pull request URL is for #{user}/#{repo} but `--tap=#{tap.full_name}` was specified!"
      end

      ohai "Dispatching #{tap} pull request ##{issue}"
      GitHub.workflow_dispatch_event(user, repo, workflow, ref, inputs)
    end
  end
end
