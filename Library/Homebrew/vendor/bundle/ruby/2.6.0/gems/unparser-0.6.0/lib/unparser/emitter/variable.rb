# frozen_string_literal: true

module Unparser
  class Emitter

    # Emitter for various variable accesses
    class Variable < self
      handle :ivar, :lvar, :cvar, :gvar, :back_ref

      children :name

    private

      def dispatch
        write(name.to_s)
      end

    end # Access

    # Emitter for constant access
    class Const < self
      handle :const

      children :scope, :name

    private

      def dispatch
        emit_scope
        write(name.to_s)
      end

      def emit_scope
        return unless scope

        visit(scope)
        write('::') unless n_cbase?(scope)
      end
    end

    # Emitter for nth_ref nodes (regexp captures)
    class NthRef < self
      PREFIX = '$'.freeze
      handle :nth_ref

      children :name

    private

      def dispatch
        write(PREFIX)
        write(name.to_s)
      end

    end # NthRef

  end # Emitter
end # Unparser
