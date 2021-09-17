# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for case nodes
    class Case < self
      handle :case

      children :condition
      define_group :whens, 1..-2

    private

      def dispatch
        write('case')
        emit_condition
        emit_whens
        emit_else
        k_end
      end

      def emit_else
        else_branch = children.last
        return unless else_branch

        write('else')
        emit_body(else_branch)
      end

      def emit_whens
        nl
        whens.each(&method(:visit))
      end

      def emit_condition
        return unless condition

        ws
        visit(condition)
      end
    end # Case

    # Emitter for when nodes
    class When < self
      handle :when

      define_group :captures, 0..-2

    private

      def dispatch
        write('when ')
        emit_captures
        emit_optional_body(children.last)
      end

      def emit_captures
        delimited(captures)
      end

    end # When
  end # Emitter
end # Unparser
