# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for in pattern nodes
    class MatchAs < self

      handle :match_as

      children :left, :right

    private

      def dispatch
        visit(left)
        write(' => ')
        visit(right)
      end
    end # MatchAs
  end # Emitter
end # Unparser
