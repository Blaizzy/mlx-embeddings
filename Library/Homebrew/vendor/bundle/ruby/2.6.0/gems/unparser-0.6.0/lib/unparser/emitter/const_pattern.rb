# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for const pattern node
    class ConstPattern < self

      handle :const_pattern

      children :const, :pattern

    private

      def dispatch
        visit(const)
        if n_hash_pattern?(pattern)
          emitter(pattern).emit_const_pattern
        else
          visit(pattern)
        end
      end
    end # ConstPattern
  end # Emitter
end # Unparser
