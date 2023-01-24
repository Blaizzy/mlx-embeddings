# typed: true
# frozen_string_literal: true

module Homebrew
  remove_const(:DEFAULT_PREFIX)
  DEFAULT_PREFIX = HOMEBREW_LINUX_DEFAULT_PREFIX

  remove_const(:DEFAULT_REPOSITORY)
  DEFAULT_REPOSITORY = HOMEBREW_LINUX_DEFAULT_REPOSITORY
end
