# frozen_string_literal: true

module Unparser
  module Writer
    class Send
      # Writer for "conditional" receiver&.selector(arguments...) case
      class Conditional < self

      private

        def dispatch
          emit_receiver
          emit_selector
          emit_arguments
        end

        def emit_receiver
          visit(receiver)
          write('&.')
        end

      end # Regular
    end # Send
  end # Writer
end # Unparser
