# typed: strict
# frozen_string_literal: true

require "cli/parser"

parser = Homebrew::CLI::Parser.new do
  usage_banner <<~EOS
    `verify-formula-undefined`

    Verifies that `require "formula"` has not been performed at startup.
  EOS
end

parser.parse

Homebrew.failed = defined?(Formula) && Formula.respond_to?(:[])
