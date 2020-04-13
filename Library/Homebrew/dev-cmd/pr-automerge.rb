# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  module_function

  def pr_automerge_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pr-automerge` [<options>]

        Finds pull requests that can be automatically merged using `brew pr-publish`.
      EOS
      flag "--tap=",
           description: "Target repository tap (default: `homebrew/core`)"
      flag "--with-label=",
           description: "Pull requests must have this label (default: `ready to merge`)"
      flag "--without-label=",
           description: "Pull requests must not have this label (default: `do not merge`)"
      switch "--publish",
             description: "Run `brew pr-publish` on matching pull requests."
      switch "--ignore-failures",
             description: "Include pull requests that have failing status checks."
      switch :debug
      switch :verbose
    end
  end

  def pr_automerge
    pr_automerge_args.parse

    ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"] = "1" unless OS.mac?
    with_label = Homebrew.args.with_label || "ready to merge"
    without_label = Homebrew.args.without_label || "do not merge"
    tap = Tap.fetch(Homebrew.args.tap || CoreTap.instance.name)

    query = "is:pr is:open repo:#{tap.full_name} label:\"#{with_label}\" -label:\"#{without_label}\""
    query += args.ignore_failures? ? " -status:pending" : " status:success"
    odebug "Searching: #{query}"

    prs = GitHub.search_issues query
    if prs.blank?
      ohai "No matching pull requests!"
      return
    end

    ohai "#{prs.size} matching pull requests:"
    pr_urls = []
    prs.each do |pr|
      puts "#{tap.full_name unless tap.core_tap?}##{pr["number"]}: #{pr["title"]}"
      pr_urls << pr["html_url"]
    end

    if args.publish?
      safe_system "#{HOMEBREW_PREFIX}/bin/brew", "pr-publish", *pr_urls
    else
      ohai "Now run:"
      puts "  brew pr-publish \\\n    #{pr_urls.join " \\\n    "}"
    end
  end
end
