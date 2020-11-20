# typed: true
# frozen_string_literal: true

module Cask
  # Helper functions for the cask cache.
  #
  # @api private
  module Cache
    extend T::Sig

    sig { returns(Pathname) }
    def self.path
      @path ||= HOMEBREW_CACHE/"Cask"
    end
  end
end
