# frozen_string_literal: true

require "commands"
require "cli/parser"

module Homebrew
  module_function

  def command_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `command` <cmd>

        Display the path to the file being used when invoking `brew` <cmd>.
      EOS
      switch :verbose
      switch :debug
    end
  end

  def command
    command_args.parse

    raise UsageError, "This command requires a command argument" if args.remaining.empty?

    args.remaining.each do |cmd|
      path = Commands.path(cmd)
      odie "Unknown command: #{cmd}" unless path
      puts path
    end
  end
end
