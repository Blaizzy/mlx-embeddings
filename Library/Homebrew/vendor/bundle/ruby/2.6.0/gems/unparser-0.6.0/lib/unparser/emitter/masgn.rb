# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for multiple assignment nodes
    class MASGN < self
      handle :masgn

      children :target, :source

    private

      def dispatch
        visit(target)
        write(' = ')
        visit(source)
      end
    end # MLHS
  end # Emitter
end # Unaprser
