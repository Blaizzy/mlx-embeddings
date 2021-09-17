# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for multiple assignment left hand side
    class MLHS < self
      handle :mlhs

      NO_COMMA = %i[arg splat mlhs restarg].freeze

      private_constant(*constants(false))

    private

      def dispatch
        if children.one?
          emit_one_child_mlhs
        else
          emit_many
        end
      end

      def emit_one_child_mlhs
        child = children.first
        parentheses do
          emitter(child).emit_mlhs
          write(',') unless NO_COMMA.include?(child.type)
        end
      end

      def emit_many
        parentheses do
          delimited(children) do |node|
            emitter(node).emit_mlhs
          end
        end
      end
    end # MLHS
  end # Emitter
end # Unaprser
