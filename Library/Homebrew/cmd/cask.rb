# frozen_string_literal: true

require "cask"

module Homebrew
  module_function

  def cask
    Cask::Cmd.run(*ARGV)
  end
end
