# typed: false
# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def pr_automerge_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Find pull requests that can be automatically merged using `brew pr-publish`.
      EOS
      flag   "--tap=",
             description: "Target tap repository (default: `homebrew/core`)."
      flag   "--with-label=",
             description: "Pull requests must have this label."
      comma_array "--without-labels",
                  description: "Pull requests must not have these labels (default: "\
                               "`do not merge`, `new formula`, `automerge-skip`, `linux-only`)."
      switch "--without-approval",
             description: "Pull requests do not require approval to be merged."
      switch "--publish",
             description: "Run `brew pr-publish` on matching pull requests."
      switch "--autosquash",
             description: "Instruct `brew pr-publish` to automatically reformat and reword commits "\
                          "in the pull request to our preferred format."
      switch "--ignore-failures",
             description: "Include pull requests that have failing status checks."

      named_args :none
    end
  end

  def pr_automerge
    args = pr_automerge_args.parse

    without_labels = args.without_labels || ["do not merge", "new formula", "automerge-skip", "linux-only"]
    tap = Tap.fetch(args.tap || CoreTap.instance.name)

    query = "is:pr is:open repo:#{tap.full_name} draft:false"
    query += args.ignore_failures? ? " -status:pending" : " status:success"
    query += " review:approved" unless args.without_approval?
    query += " label:\"#{args.with_label}\"" if args.with_label
    without_labels&.each { |label| query += " -label:\"#{label}\"" }
    odebug "Searching: #{query}"

    prs = GitHub.search_issues query
    if prs.blank?
      ohai "No matching pull requests!"
      return
    end

    ohai "#{prs.count} matching pull #{"request".pluralize(prs.count)}:"
    pr_urls = []
    prs.each do |pr|
      puts "#{tap.full_name unless tap.core_tap?}##{pr["number"]}: #{pr["title"]}"
      pr_urls << pr["html_url"]
    end

    publish_args = ["pr-publish"]
    publish_args << "--tap=#{tap}" if tap
    publish_args << "--autosquash" if args.autosquash?
    if args.publish?
      safe_system HOMEBREW_BREW_FILE, *publish_args, *pr_urls
    else
      ohai "Now run:", "  brew #{publish_args.join " "} \\\n    #{pr_urls.join " \\\n    "}"
    end
  end
end
