# frozen_string_literal: true

require "system_config"
require "cli/parser"

module Homebrew
  module_function

  def config_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `config`

        Show Homebrew and system configuration info useful for debugging. If you file
        a bug report, you will be required to provide this information.
      EOS
      switch :verbose
      switch :debug
    end
  end

  def config
    config_args.parse
    raise UsageError unless args.remaining.empty?

    SystemConfig.dump_verbose_config
  end
end
