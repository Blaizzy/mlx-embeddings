# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for send to index references
    class Index < self

    private

      def dispatch
        emit_receiver
        emit_operation
      end

      def emit_receiver
        visit(first_child)
      end

      class Reference < self
        define_group(:indices, 1..-1)

        handle :index

      private

        def emit_operation
          parentheses('[', ']') do
            delimited(indices)
          end
        end
      end # Reference

      # Emitter for assign to index nodes
      class Assign < self

        handle :indexasgn

        VALUE_RANGE     = (1..-2).freeze
        NO_VALUE_PARENT = %i[and_asgn op_asgn or_asgn].to_set.freeze

        private_constant(*constants(false))

        def emit_heredoc_reminders
          emitter(children.last).emit_heredoc_reminders
        end

        def dispatch
          emit_receiver
          emit_operation(children[VALUE_RANGE])
          write(' = ')
          visit(children.last)
        end

        def emit_mlhs
          emit_receiver
          emit_operation(children.drop(1))
        end

      private

        def emit_operation(indices)
          parentheses('[', ']') do
            delimited(indices)
          end
        end
      end # Assign
    end # Index
  end # Emitter
end # Unparser
