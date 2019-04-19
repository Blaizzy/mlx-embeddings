# frozen_string_literal: true

module Homebrew
  DEFAULT_PREFIX ||= if ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"]
    "/usr/local"
  else
    "/home/linuxbrew/.linuxbrew"
  end.freeze
end
