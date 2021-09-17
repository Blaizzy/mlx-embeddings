# frozen_string_literal: true

module Unparser
  class Emitter
    # Base class for primitive emitters
    class Primitive < self

      children :value

      # Emitter for primitives based on Object#inspect
      class Inspect < self

        handle :sym, :str

      private

        def dispatch
          write(value.inspect)
        end

      end # Inspect

      # Emitter for complex literals
      class Complex < self

        handle :complex

        RATIONAL_FORMAT = 'i'.freeze

        MAP =
          {
            ::Float    => :float,
            ::Rational => :rational,
            ::Integer  => :int
          }.freeze

      private

        def dispatch
          emit_imaginary
          write(RATIONAL_FORMAT)
        end

        def emit_imaginary
          visit(imaginary_node)
        end

        def imaginary_node
          imaginary = value.imaginary
          s(MAP.fetch(imaginary.class), imaginary)
        end

      end # Rational

      # Emitter for rational literals
      class Rational < self

        handle :rational

        RATIONAL_FORMAT = 'r'.freeze

      private

        # rubocop:disable Lint/FloatComparison
        def dispatch
          integer = Integer(value)
          float   = value.to_f

          write_rational(integer.to_f.equal?(float) ? integer : float)
        end
        # rubocop:enable Lint/FloatComparison

        def write_rational(value)
          write(value.to_s, RATIONAL_FORMAT)
        end

      end # Rational

      # Emiter for numeric literals
      class Numeric < self

        handle :int

      private

        def dispatch
          write(value.inspect)
        end

      end # Numeric
    end # Primitive
  end # Emitter
end # Unparser
