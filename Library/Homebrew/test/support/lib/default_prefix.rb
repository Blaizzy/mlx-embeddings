# typed: true
# frozen_string_literal: true

module Homebrew
  # For testing's sake always assume the default prefix
  DEFAULT_PREFIX = HOMEBREW_PREFIX.to_s.freeze
  DEFAULT_REPOSITORY = HOMEBREW_REPOSITORY.to_s.freeze
end
