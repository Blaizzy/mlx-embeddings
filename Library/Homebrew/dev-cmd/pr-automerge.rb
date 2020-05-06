# frozen_string_literal: true

require "cli/parser"
require "utils/github"

module Homebrew
  module_function

  def pr_automerge_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `pr-automerge` [<options>]

        Find pull requests that can be automatically merged using `brew pr-publish`.
      EOS
      flag   "--tap=",
             description: "Target tap repository (default: `homebrew/core`)."
      flag   "--with-label=",
             description: "Pull requests must have this label."
      comma_array "--without-labels=",
                  description: "Pull requests must not have these labels (default: `do not merge`, `new formula`)."
      switch "--without-approval",
             description: "Pull requests do not require approval to be merged."
      switch "--publish",
             description: "Run `brew pr-publish` on matching pull requests."
      switch "--ignore-failures",
             description: "Include pull requests that have failing status checks."
      switch :verbose
      switch :debug
      max_named 0
    end
  end

  def pr_automerge
    pr_automerge_args.parse

    ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"] = "1" unless OS.mac?
    without_labels = Homebrew.args.without_labels || ["do not merge", "new formula"]
    tap = Tap.fetch(Homebrew.args.tap || CoreTap.instance.name)

    query = "is:pr is:open repo:#{tap.full_name}"
    query += Homebrew.args.ignore_failures? ? " -status:pending" : " status:success"
    query += " review:approved" unless Homebrew.args.without_approval?
    query += " label:\"#{with_label}\"" if Homebrew.args.with_label
    without_labels&.each { |label| query += " -label:\"#{label}\"" }
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
      ohai "Now run:", "  brew pr-publish \\\n    #{pr_urls.join " \\\n    "}"
    end
  end
end
