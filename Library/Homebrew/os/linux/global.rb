# typed: false
# frozen_string_literal: true

require "env_config"

# Enables `patchelf.rb` write support.
HOMEBREW_PATCHELF_RB_WRITE = ENV["HOMEBREW_NO_PATCHELF_RB_WRITE"].blank?.freeze

module Homebrew
  if EnvConfig.force_homebrew_on_linux?
    DEFAULT_PREFIX ||= HOMEBREW_DEFAULT_PREFIX.freeze
    DEFAULT_REPOSITORY ||= HOMEBREW_DEFAULT_REPOSITORY.freeze
  else
    DEFAULT_PREFIX ||= HOMEBREW_LINUX_DEFAULT_PREFIX.freeze
    DEFAULT_REPOSITORY ||= HOMEBREW_LINUX_DEFAULT_REPOSITORY.freeze
  end
end
