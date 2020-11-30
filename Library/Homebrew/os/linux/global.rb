# typed: false
# frozen_string_literal: true

require "env_config"

# Enables experimental `patchelf.rb` write support.
HOMEBREW_PATCHELF_RB_WRITE = (
  ENV["HOMEBREW_NO_PATCHELF_RB_WRITE"].blank? &&
  (ENV["HOMEBREW_PATCHELF_RB_WRITE"].present? || ENV["CI"].present? || ENV["HOMEBREW_DEV_CMD_RUN"].present?)
).freeze

module Homebrew
  DEFAULT_PREFIX ||= if EnvConfig.force_homebrew_on_linux?
    HOMEBREW_DEFAULT_PREFIX
  else
    HOMEBREW_LINUX_DEFAULT_PREFIX
  end.freeze
end
