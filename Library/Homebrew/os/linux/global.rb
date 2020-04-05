# frozen_string_literal: true

module Homebrew
  DEFAULT_PREFIX ||= if Homebrew::EnvConfig.force_homebrew_on_linux?
    HOMEBREW_DEFAULT_PREFIX
  else
    LINUXBREW_DEFAULT_PREFIX
  end.freeze
end
