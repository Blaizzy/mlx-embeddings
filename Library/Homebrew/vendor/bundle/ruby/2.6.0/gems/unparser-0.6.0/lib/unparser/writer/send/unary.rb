# frozen_string_literal: true

module Unparser
  module Writer
    class Send
      # Writer for unary sends
      class Unary < self
        MAP = {
          '-@': '-',
          '+@': '+'
        }.freeze

        private_constant(*constants(false))

        def dispatch
          name = selector

          write(MAP.fetch(name, name).to_s)

          if n_int?(receiver) && selector.equal?(:'+@')
            write('+')
          end

          visit(receiver)
        end
      end # Unary
    end # Send
  end # Writer
end # Unparser
