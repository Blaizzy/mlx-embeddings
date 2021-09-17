# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for array patterns
    class ArrayPattern < self

      handle :array_pattern
      handle :array_pattern_with_tail

    private

      def dispatch
        write('[')
        delimited(children, &method(:emit_member))
        write(', ') if node_type.equal?(:array_pattern_with_tail)
        write(']')
      end

      def emit_member(node)
        if n_match_rest?(node)
          writer_with(MatchRest, node).emit_array_pattern
        else
          visit(node)
        end
      end
    end # Pin
  end # Emitter
end # Unparser
