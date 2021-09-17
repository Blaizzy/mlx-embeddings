# frozen_string_literal: true

module Unparser
  class Emitter
    # Emiter for float literals
    class Float < self
      handle :float

      children :value

      INFINITY     = ::Float::INFINITY
      NEG_INFINITY = -::Float::INFINITY

    private

      def dispatch
        case value
        when INFINITY
          write('10e1000000000000000000')
        when NEG_INFINITY
          write('-10e1000000000000000000')
        else
          write(value.inspect)
        end
      end

    end # Float
  end # Emitter
end # Unparser
