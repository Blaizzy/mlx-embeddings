# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for toplevel constant reference nodes
    class CBase < self
      handle :cbase

    private

      # Perform dispatch
      #
      # @return [undefined]
      #
      # @api private
      #
      def dispatch
        write('::')
      end

    end # CBase
  end # Emitter
end # Unparser
