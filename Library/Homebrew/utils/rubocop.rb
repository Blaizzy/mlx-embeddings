#!/usr/bin/env ruby
# typed: false
# frozen_string_literal: true

require_relative "gems"
Homebrew.setup_gem_environment!

require_relative "../warnings"

Warnings.ignore :parser_syntax do
  require "rubocop"
end

exit RuboCop::CLI.new.run
