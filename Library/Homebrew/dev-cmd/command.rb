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

      min_named 1
    end
  end

  def command
    args = command_args.parse

    args.named.each do |cmd|
      path = Commands.path(cmd)
      odie "Unknown command: #{cmd}" unless path
      puts path
    end
  end
end
