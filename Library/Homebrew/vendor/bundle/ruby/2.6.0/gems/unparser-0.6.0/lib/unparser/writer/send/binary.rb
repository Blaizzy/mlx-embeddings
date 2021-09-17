# frozen_string_literal: true

module Unparser
  module Writer
    class Send
      # Writer for binary sends
      class Binary < self
        def dispatch
          visit(receiver)
          emit_operator
          emit_right
        end

      private

        def emit_operator
          write(' ', details.string_selector, ' ')
        end

        def emit_right
          emit_send_regular(children.fetch(2))
        end

      end # Binary
    end # Send
  end # Writer
end # Unparser
