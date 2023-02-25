# typed: true
# frozen_string_literal: true

module Homebrew
  DEFAULT_PREFIX = ENV.fetch("HOMEBREW_DEFAULT_PREFIX").freeze
  DEFAULT_REPOSITORY = ENV.fetch("HOMEBREW_DEFAULT_REPOSITORY").freeze
end
