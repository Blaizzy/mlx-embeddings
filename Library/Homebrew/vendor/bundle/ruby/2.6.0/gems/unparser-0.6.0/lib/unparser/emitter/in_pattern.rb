# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for in pattern nodes
    class InPattern < self

      handle :in_pattern

      children :target, :unless_guard, :branch, :else_branch

    private

      def dispatch
        write('in')

        ws

        visit(target)

        if unless_guard
          ws
          visit(unless_guard)
        end

        if branch
          ws
          write('then')
          emit_body(branch)
        else
          nl
        end
      end
    end # InPattern
  end # Emitter
end # Unparser
