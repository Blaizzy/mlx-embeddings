# typed: false
# frozen_string_literal: true

require "optparse"
require "shellwords"

require "cli/parser"
require "extend/optparse"

require "cask/config"

require "cask/cmd/abstract_command"
require "cask/cmd/audit"
require "cask/cmd/fetch"
require "cask/cmd/info"
require "cask/cmd/install"
require "cask/cmd/list"
require "cask/cmd/reinstall"
require "cask/cmd/uninstall"
require "cask/cmd/upgrade"
require "cask/cmd/zap"

module Cask
  # Implementation of the `brew cask` command-line interface.
  #
  # @api private
  class Cmd
    extend T::Sig

    include Context

    def self.parser(&block)
      Homebrew::CLI::Parser.new do
        instance_eval(&block) if block

        cask_options
      end
    end
  end
end
