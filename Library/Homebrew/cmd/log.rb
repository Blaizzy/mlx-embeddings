# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  module_function

  def log_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `log` [<options>] [<formula>]

        Show the `git log` for <formula>, or show the log for the Homebrew repository
        if no formula is provided.
      EOS
      switch "-p", "-u", "--patch",
             description: "Also print patch from commit."
      switch "--stat",
             description: "Also print diffstat from commit."
      switch "--oneline",
             description: "Print only one line per commit."
      flag   "-1", "--max-count",
             description: "Print only one or a specified number of commits."
      max_named 1
    end
  end

  def log
    log_args.parse

    # As this command is simplifying user run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    if args.no_named?
      git_log HOMEBREW_REPOSITORY
    else
      path = Formulary.path(args.named.first)
      tap = Tap.from_path(path)
      git_log path.dirname, path, tap
    end
  end

  def git_log(cd_dir, path = nil, tap = nil)
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
        To get a full clone run:
          git -C "#{git_cd}" fetch --unshallow
      EOS
    end
    system_args = args.options_only
    system_args += ["--follow", "--", path] if path.present?
    system "git", "log", *system_args
  end
end
