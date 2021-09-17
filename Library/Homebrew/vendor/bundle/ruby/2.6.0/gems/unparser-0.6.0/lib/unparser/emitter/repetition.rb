# frozen_string_literal: true

module Unparser
  class Emitter

    # Emitter for postconditions
    class Post < self
      children :condition, :body

      MAP = {
        while_post: 'while',
        until_post: 'until'
      }.freeze

      handle(*MAP.keys)

    private

      def dispatch
        visit(body)
        write(' ', MAP.fetch(node.type), ' ')
        visit(condition)
      end
    end

    # Emitter for while and until nodes
    class Repetition < self
      MAP = {
        while: 'while',
        until: 'until'
      }.freeze

      handle(*MAP.keys)

      children :condition, :body

    private

      def dispatch
        if postcontrol?
          emit_postcontrol
        else
          emit_normal
        end
      end

      def postcontrol?
        body && local_variable_scope.first_assignment_in?(body, condition)
      end

      def emit_keyword
        write(MAP.fetch(node.type), ' ')
      end

      def emit_normal
        emit_keyword
        visit(condition)
        if body
          emit_body(body)
        else
          nl
        end
        k_end
      end

      def emit_postcontrol
        visit(body)
        ws
        emit_keyword
        visit(condition)
      end

    end # Repetition
  end # Emitter
end # Unparser
