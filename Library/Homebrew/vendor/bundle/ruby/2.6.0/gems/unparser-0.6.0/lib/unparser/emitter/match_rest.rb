# frozen_string_literal: true

module Unparser
  class Emitter
    # Emiter for match rest nodes
    class MatchRest < self
      children :match_var

      def emit_array_pattern
        write('*')
        emit_match_var
      end

      def emit_hash_pattern
        write('**')
        emit_match_var
      end

    private

      def emit_match_var
        visit(match_var) if match_var
      end
    end # MatchRest
  end # Emitter
end # Unparser
