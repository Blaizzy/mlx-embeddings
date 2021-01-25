# typed: true
# frozen_string_literal: true

require "cli/parser"
require "release_notes"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def release_notes_args
    Homebrew::CLI::Parser.new do
      usage_banner "`release-notes` [<options>] [<previous_tag>] [<end_ref>]"
      description <<~EOS
        Print the merged pull requests on Homebrew/brew between two Git refs.
        If no <previous_tag> is provided it defaults to the latest tag.
        If no <end_ref> is provided it defaults to `origin/master`.

        If `--markdown` and a <previous_tag> are passed, an extra line containing
        a link to the Homebrew blog will be adding to the output. Additionally,
        a warning will be shown if the latest minor release was less than one month ago.
      EOS
      switch "--markdown",
             description: "Print as a Markdown list."

      named_args max: 2

      hide_from_man_page!
    end
  end

  def release_notes
    args = release_notes_args.parse

    odeprecated "`brew release-notes`", "`brew release`"

    previous_tag = args.named.first

    if previous_tag.present?
      most_recent_major_minor_tag = previous_tag.sub(/\d+$/, "0")
      one_month_ago = Date.today << 1
      previous_tag_date = Date.parse Utils.popen_read(
        "git", "-C", HOMEBREW_REPOSITORY, "log", "-1", "--format=%aI", most_recent_major_minor_tag
      )
      opoo "The latest major/minor release was less than one month ago." if previous_tag_date > one_month_ago
    end

    previous_tag ||= Utils.popen_read(
      "git", "-C", HOMEBREW_REPOSITORY, "tag", "--list", "--sort=-version:refname"
    ).lines.first.chomp
    odie "Could not find any previous tags!" unless previous_tag

    end_ref = args.named.second || "origin/master"

    [previous_tag, end_ref].each do |ref|
      next if quiet_system "git", "-C", HOMEBREW_REPOSITORY, "rev-parse", "--verify", "--quiet", ref

      odie "Ref #{ref} does not exist!"
    end

    release_notes = ReleaseNotes.generate_release_notes previous_tag, end_ref, markdown: args.markdown?

    $stderr.puts "Release notes between #{previous_tag} and #{end_ref}:"
    puts release_notes
  end
end
