#!/usr/bin/env ruby
# typed: false
# frozen_string_literal: true

require "warning"

warnings = [
  "warning: parser/current is loading parser/ruby26, which recognizes",
  "warning: 2.6.6-compliant syntax, but you are running 2.6.3.",
  "warning: please see https://github.com/whitequark/parser#compatibility-with-ruby-mri.",
]

warnings.each do |warning|
  Warning.ignore Regexp.new(warning)
end

require "rubocop"

exit RuboCop::CLI.new.run
