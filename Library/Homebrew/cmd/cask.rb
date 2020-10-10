# typed: false
# frozen_string_literal: true

require "cask/cmd"

module Homebrew
  module_function

  def cask_args
    Cask::Cmd.parser
  end

  def cask
    ARGV.freeze
    Cask::Cmd.run(*ARGV)
  end
end
