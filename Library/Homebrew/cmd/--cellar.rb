# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def __cellar_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display Homebrew's Cellar path. *Default:* `$(brew --prefix)/Cellar`, or if
        that directory doesn't exist, `$(brew --repository)/Cellar`.

        If <formula> is provided, display the location in the Cellar where <formula>
        would be installed, without any sort of versioned directory as the last path.
      EOS

      named_args :formula
    end
  end

  def __cellar
    args = __cellar_args.parse

    if args.no_named?
      puts HOMEBREW_CELLAR
    else
      puts args.named.to_resolved_formulae.map(&:rack)
    end
  end
end
