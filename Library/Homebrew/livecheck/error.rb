# typed: true
# frozen_string_literal: true

module Homebrew
  module Livecheck
    # Error during a livecheck run.
    #
    # @api private
    class Error < RuntimeError
    end
  end
end
