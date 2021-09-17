# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for in pattern nodes
    class MatchAlt < self

      handle :match_alt

      children :left, :right

    private

      def dispatch
        visit(left)
        ws
        write('|')
        ws
        visit(right)
      end
    end # MatchAlt
  end # Emitter
end # Unparser
