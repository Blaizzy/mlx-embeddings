require "formula"
require "cli/parser"

module Homebrew
  module_function

  def log_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `log` [<options>] <formula>

        Show the `git log` for the given <formula>.
      EOS
      switch "-p", "-u", "--patch",
        description: "Also output patch from commit."
      switch "--stat",
        description: "Also output diffstat from commit."
      switch "--oneline",
        description: "Output only one line per commit."
      switch "-1", "--max-count=1",
        description: "Output only one commit."
    end
  end

  def log
    log_args.parse

    if ARGV.named.empty?
      git_log HOMEBREW_REPOSITORY
    else
      path = Formulary.path(ARGV.named.first)
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
    args = ARGV.options_only
    args += ["--follow", "--", path] unless path.nil?
    system "git", "log", *args
  end
end
