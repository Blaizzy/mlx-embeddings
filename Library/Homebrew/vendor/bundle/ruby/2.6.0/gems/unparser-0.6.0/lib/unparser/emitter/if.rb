# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter if nodes
    class If < self
      handle :if

      children :condition, :if_branch, :else_branch

      def emit_ternary
        visit(condition)
        write(' ? ')
        visit(if_branch)
        write(' : ')
        visit(else_branch)
      end

    private

      def dispatch
        if postcondition?
          emit_postcondition
        else
          emit_normal
        end
      end

      def postcondition?
        return false unless if_branch.nil? ^ else_branch.nil?

        body = if_branch || else_branch

        local_variable_scope.first_assignment_in?(body, condition)
      end

      def emit_postcondition
        visit(if_branch || else_branch)
        write(' ', keyword, ' ')
        emit_condition
      end

      def emit_normal
        write(keyword, ' ')
        emit_condition
        emit_if_branch
        emit_else_branch
        k_end
      end

      def unless?
        !if_branch && else_branch
      end

      def keyword
        unless? ? 'unless' : 'if'
      end

      def emit_condition
        visit(condition)
      end

      def emit_if_branch
        if if_branch
          emit_body(if_branch)
        end

        nl if !if_branch && !else_branch
      end

      def emit_else_branch
        return unless else_branch

        write('else') unless unless?
        emit_body(else_branch)
      end

    end # If
  end # Emitter
end # Unparser
