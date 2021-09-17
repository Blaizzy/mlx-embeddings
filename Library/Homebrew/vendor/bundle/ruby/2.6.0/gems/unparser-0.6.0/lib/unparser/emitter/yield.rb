# frozen_string_literal: true

module Unparser
  class Emitter

    # Emitter for yield node
    class Yield < self
      handle :yield

    private

      def dispatch
        write('yield')
        return if children.empty?

        parentheses do
          delimited(children)
        end
      end

    end # Yield

  end # Emitter
end # Unparser
