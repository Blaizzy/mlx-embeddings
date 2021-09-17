# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for case guards
    class CaseGuard < self

      handle :if_guard, :unless_guard

      MAP = {
        if_guard:     'if',
        unless_guard: 'unless'
      }.freeze

      children :condition

    private

      def dispatch
        write(MAP.fetch(node_type))
        ws
        visit(condition)
      end

    end # UnlessGuard
  end # Emitter
end # Unparser
