# typed: true
# frozen_string_literal: true

require "system_config"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def config_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Show Homebrew and system configuration info useful for debugging. If you file
        a bug report, you will be required to provide this information.
      EOS

      named_args :none
    end
  end

  def config
    config_args.parse

    SystemConfig.dump_verbose_config
  end
end
