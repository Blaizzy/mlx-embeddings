# frozen_string_literal: true

require "net/http"
require "net/https"
require "json"
require "cli/parser"
require "formula"
require "formulary"
require "version"
require "pkg_version"
require "formula_info"

module Homebrew
  module_function

  def pull_args
    Homebrew::CLI::Parser.new do
      hide_from_man_page!
      usage_banner <<~EOS
        `pull` [<options>] <patch>

        Get a patch from a GitHub commit or pull request and apply it to Homebrew.

        Each <patch> may be the number of a pull request in `homebrew/core`
        or the URL of any pull request or commit on GitHub.
      EOS
      switch "--bump",
             description: "For one-formula PRs, automatically reword commit message to our preferred format."
      switch "--clean",
             description: "Do not rewrite or otherwise modify the commits found in the pulled PR."
      switch "--ignore-whitespace",
             description: "Silently ignore whitespace discrepancies when applying diffs."
      switch "--resolve",
             description: "When a patch fails to apply, leave in progress and allow user to resolve, instead "\
                          "of aborting."
      switch "--branch-okay",
             description: "Do not warn if pulling to a branch besides master (useful for testing)."
      switch "--no-pbcopy",
             description: "Do not copy anything to the system clipboard."

      min_named 1
    end
  end

  def pull
    odisabled "brew pull", "gh pr checkout"
  end
end
