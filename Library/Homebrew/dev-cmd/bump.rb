require "cli/parser"

module Homebrew
  module_function

  def bump_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump`

        Display out-of-date brew formulae, the latest version available, and whether a pull request has been opened.
      EOS
    end
  end

  def bump
    bump_args.parse
  end
end
