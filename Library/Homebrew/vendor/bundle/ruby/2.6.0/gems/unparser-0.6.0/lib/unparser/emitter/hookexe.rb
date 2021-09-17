# frozen_string_literal: true

module Unparser
  class Emitter
    # Base class for pre and postexe emitters
    class Hookexe < self

      MAP = {
        preexe:  'BEGIN',
        postexe: 'END'
      }.freeze

      handle(*MAP.keys)

      children :body

    private

      def dispatch
        write(MAP.fetch(node.type), ' ')
        parentheses('{', '}') do
          emit_body(body)
        end
      end

    end # Hookexe
  end # Emitter
end # Unparser
