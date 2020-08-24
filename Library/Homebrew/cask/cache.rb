# frozen_string_literal: true

module Cask
  # Helper functions for the cask cache.
  #
  # @api private
  module Cache
    module_function

    def path
      @path ||= HOMEBREW_CACHE/"Cask"
    end
  end
end
