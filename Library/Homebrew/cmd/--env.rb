# frozen_string_literal: true

require "extend/ENV"
require "build_environment"
require "utils/shell"
require "cli/parser"

module Homebrew
  module_function

  def __env_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--env` [<options>] [<formula>]

        Summarise Homebrew's build environment as a plain list.

        If the command's output is sent through a pipe and no shell is specified,
        the list is formatted for export to `bash`(1) unless `--plain` is passed.
      EOS
      flag   "--shell=",
             description: "Generate a list of environment variables for the specified shell, " \
                          "or `--shell=auto` to detect the current shell."
      switch "--plain",
             description: "Generate plain output even when piped."
    end
  end

  def __env
    __env_args.parse

    ENV.activate_extensions!
    ENV.deps = Homebrew.args.formulae if superenv?
    ENV.setup_build_environment
    ENV.universal_binary if ARGV.build_universal?

    shell = if args.plain?
      nil
    elsif args.shell.nil?
      # legacy behavior
      :bash unless $stdout.tty?
    elsif args.shell == "auto"
      Utils::Shell.parent || Utils::Shell.preferred
    elsif args.shell
      Utils::Shell.from_path(args.shell)
    end

    env_keys = build_env_keys(ENV)
    if shell.nil?
      dump_build_env ENV
    else
      env_keys.each do |key|
        puts Utils::Shell.export_value(key, ENV[key], shell)
      end
    end
  end
end
