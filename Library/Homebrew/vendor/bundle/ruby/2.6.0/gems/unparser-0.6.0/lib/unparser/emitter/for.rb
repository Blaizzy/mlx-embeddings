# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for for nodes
    class For < self
      handle :for

      children :condition, :assignment, :body

    private

      def dispatch
        write('for ')
        emit_condition
        emit_optional_body(body)
        k_end
      end

      def emit_condition
        visit(condition)
        write(' in ')
        visit(assignment)
        write(' do')
      end

    end # For
  end # Emitter
end # Unparser
