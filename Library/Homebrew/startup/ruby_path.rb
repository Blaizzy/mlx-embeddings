# typed: true
# frozen_string_literal: true

require "rbconfig"

RUBY_PATH = Pathname.new(RbConfig.ruby).freeze
RUBY_BIN = RUBY_PATH.dirname.freeze
