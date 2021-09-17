# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for in pattern nodes
    class MatchPattern < self

      handle :match_pattern
      handle :match_pattern_p

      children :target, :pattern

    private

      def dispatch
        visit(target)
        write(' in ')
        visit(pattern)
      end
    end # InPattern
  end # Emitter
end # Unparser
