# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for in pattern nodes
    class InMatch < self

      handle :in_match

      children :target, :pattern

    private

      def dispatch
        visit(target)
        write(' in ')
        visit(pattern)
      end
    end # InMatch
  end # Emitter
end # Unparser
