# frozen_string_literal: true

module Homebrew
  DEFAULT_PREFIX ||= if ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"]
    HOMEBREW_DEFAULT_PREFIX
  else
    LINUXBREW_DEFAULT_PREFIX
  end.freeze
end
