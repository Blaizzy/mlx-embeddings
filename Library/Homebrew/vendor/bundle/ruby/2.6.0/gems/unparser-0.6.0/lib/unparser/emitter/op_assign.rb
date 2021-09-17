# frozen_string_literal: true

module Unparser
  class Emitter

    # Base class for and and or op-assign
    class BinaryAssign < self
      children :target, :expression

      MAP = {
        and_asgn: '&&=',
        or_asgn:  '||='
      }.freeze

      handle(*MAP.keys)

      def emit_heredoc_reminders
        emitter(target).emit_heredoc_reminders
        emitter(expression).emit_heredoc_reminders
      end

    private

      def dispatch
        emitter(target).emit_mlhs
        write(' ', MAP.fetch(node.type), ' ')
        visit(expression)
      end

    end # BinaryAssign

    # Emitter for op assign
    class OpAssign < self
      handle :op_asgn

      children :target, :operator, :value

    private

      def dispatch
        emitter(first_child).emit_mlhs
        emit_operator
        visit(value)
      end

      def emit_operator
        write(' ', operator.to_s, '= ')
      end

    end # OpAssign
  end # Emitte
end # Unparser
