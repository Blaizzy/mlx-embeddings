#!/usr/bin/env ruby
# typed: false
# frozen_string_literal: true

require "warning"

warnings = [
  %r{warning: parser/current is loading parser/ruby\d+, which recognizes},
  /warning: \d+\.\d+\.\d+-compliant syntax, but you are running \d+\.\d+\.\d+\./,
  %r{warning: please see https://github\.com/whitequark/parser#compatibility-with-ruby-mri\.},
]

warnings.each do |warning|
  Warning.ignore warning
end

require "rubocop"

exit RuboCop::CLI.new.run
