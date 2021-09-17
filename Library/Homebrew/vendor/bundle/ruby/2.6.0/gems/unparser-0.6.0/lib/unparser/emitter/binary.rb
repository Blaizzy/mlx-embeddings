# frozen_string_literal: true

module Unparser
  class Emitter
    # Non send binary operator / keyword emitter
    class Binary < self
      handle :and, :or

    private

      def dispatch
        writer.dispatch
      end

      def writer
        writer_with(Writer::Binary, node)
      end
      memoize :writer
    end # Binary
  end # Emitter
end # Unparser
