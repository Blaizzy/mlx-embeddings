# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for defined? nodes
    class Defined < self
      handle :defined?

      children :subject

    private

      def dispatch
        write('defined?')
        parentheses { visit(subject) }
      end

    end # Defined
  end # Emitter
end # Unparser
