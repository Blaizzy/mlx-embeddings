# typed: false
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def log_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Show the `git log` for <formula>, or show the log for the Homebrew repository
        if no formula is provided.
      EOS
      switch "-p", "-u", "--patch",
             description: "Also print patch from commit."
      switch "--stat",
             description: "Also print diffstat from commit."
      switch "--oneline",
             description: "Print only one line per commit."
      switch "-1",
             description: "Print only one commit."
      flag   "-n", "--max-count=",
             description: "Print only a specified number of commits."

      conflicts "-1", "--max-count"

      named_args :formula, max: 1
    end
  end

  def log
    args = log_args.parse

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    if args.no_named?
      git_log HOMEBREW_REPOSITORY, args: args
    else
      path = Formulary.path(args.named.first)
      tap = Tap.from_path(path)
      git_log path.dirname, path, tap, args: args
    end
  end

  def git_log(cd_dir, path = nil, tap = nil, args:)
    cd cd_dir
    repo = Utils.popen_read("git rev-parse --show-toplevel").chomp
    if tap
      name = tap.to_s
      git_cd = "$(brew --repo #{tap})"
    elsif cd_dir == HOMEBREW_REPOSITORY
      name = "Homebrew/brew"
      git_cd = "$(brew --repo)"
    else
      name, git_cd = cd_dir
    end

    if File.exist? "#{repo}/.git/shallow"
      opoo <<~EOS
        #{name} is a shallow clone so only partial output will be shown.
        To get a full clone, run:
          git -C "#{git_cd}" fetch --unshallow
      EOS
    end

    git_args = []
    git_args << "--patch" if args.patch?
    git_args << "--stat" if args.stat?
    git_args << "--oneline" if args.oneline?
    git_args << "-1" if args.public_send(:'1?')
    git_args << "--max-count" << args.max_count if args.max_count
    git_args += ["--follow", "--", path] if path.present?
    system "git", "log", *git_args
  end
end
