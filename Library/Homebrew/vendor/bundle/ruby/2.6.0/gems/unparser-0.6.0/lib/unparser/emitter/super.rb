# frozen_string_literal: true

module Unparser
  class Emitter

    # Emitter for super nodes
    class Super < self
      handle :super

    private

      def dispatch
        write('super')
        parentheses do
          delimited(children)
        end
      end

    end # Super

  end # Emitter
end # Unparser
