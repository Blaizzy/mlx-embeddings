# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def __cellar_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `--cellar` [<formula>]

        Display Homebrew's Cellar path. *Default:* `$(brew --prefix)/Cellar`, or if
        that directory doesn't exist, `$(brew --repository)/Cellar`.

        If <formula> is provided, display the location in the cellar where <formula>
        would be installed, without any sort of versioned directory as the last path.
      EOS
    end
  end

  def __cellar
    __cellar_args.parse

    if Homebrew.args.named.blank?
      puts HOMEBREW_CELLAR
    else
      puts Homebrew.args.resolved_formulae.map(&:rack)
    end
  end
end
