# typed: true
# frozen_string_literal: true

require "cask/staged"

module Cask
  class DSL
    # Class corresponding to the `postflight` stanza.
    #
    # @api private
    class Postflight < Base
      include Staged

      def suppress_move_to_applications(options = {})
        # TODO: Remove from all casks because it is no longer needed
      end
    end
  end
end
