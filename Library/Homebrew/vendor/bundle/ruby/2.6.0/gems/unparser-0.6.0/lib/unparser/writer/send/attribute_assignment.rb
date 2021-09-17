# frozen_string_literal: true

module Unparser
  module Writer
    class Send
      # Writer for send as attribute assignment
      class AttributeAssignment < self
        children :receiver, :selector, :first_argument

        def dispatch
          emit_receiver
          emit_attribute
          write('=')

          if arguments.one?
            visit(first_argument)
          else
            parentheses { delimited(arguments) }
          end
        end

        def emit_send_mlhs
          emit_receiver
          write(details.non_assignment_selector)
        end

      private

        def emit_receiver
          visit(receiver)
          emit_operator
        end

        def emit_attribute
          write(details.non_assignment_selector)
        end
      end # AttributeAssignment
    end # Send
  end # Writer
end # Unparser
