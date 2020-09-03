# frozen_string_literal: true

require "cli/args"

module DependenciesHelpers
  module Compat
    def argv_includes_ignores(argv = nil)
      unless @printed_includes_ignores_warning
        odisabled "Homebrew.argv_includes_ignores", "Homebrew.args_includes_ignores"
        @printed_includes_ignores_warning = true
      end
      args_includes_ignores(argv ? Homebrew::CLI::Args.new : Homebrew.args)
    end
  end

  prepend Compat
end
