# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def release_notes_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `release-notes` [<options>] [<previous_tag>] [<end_ref>]

        Print the merged pull requests on Homebrew/brew between two Git refs.
        If no <previous_tag> is provided it defaults to the latest tag.
        If no <end_ref> is provided it defaults to `origin/master`.

        If `--markdown` and a <previous_tag> are passed, an extra line containg
        a link to the Homebrew blog will be adding to the output. Additionally,
        a warning will be shown if the latest minor release was less than one month ago.
      EOS
      switch "--markdown",
             description: "Print as a Markdown list."

      max_named 2
    end
  end

  def release_notes
    args = release_notes_args.parse

    previous_tag = args.named.first

    if previous_tag.present?

      previous_tag_date = Date.parse Utils.popen_read(
        "git", "-C", HOMEBREW_REPOSITORY, "log", "-1", "--format=%aI", previous_tag.sub(/\d+$/, "0")
      )
      opoo "The latest major/minor release was less than one month ago." if previous_tag_date > (Date.today << 1)
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

    output = Utils.popen_read(
      "git", "-C", HOMEBREW_REPOSITORY, "log", "--pretty=format:'%s >> - %b%n'", "#{previous_tag}..#{end_ref}"
    ).lines.grep(/Merge pull request/)

    output.map! do |s|
      s.gsub(%r{.*Merge pull request #(\d+) from ([^/]+)/[^>]*(>>)*},
             "https://github.com/Homebrew/brew/pull/\\1 (@\\2)")
    end
    if args.markdown?
      output.map! do |s|
        /(.*\d)+ \(@(.+)\) - (.*)/ =~ s
        "- [#{Regexp.last_match(3)}](#{Regexp.last_match(1)}) (@#{Regexp.last_match(2)})"
      end
    end

    $stderr.puts "Release notes between #{previous_tag} and #{end_ref}:"
    if args.markdown? && args.named.first
      puts "Release notes for major and minor releases can be found in the [Homebrew blog](https://brew.sh/blog/)."
    end
    puts output
  end
end
