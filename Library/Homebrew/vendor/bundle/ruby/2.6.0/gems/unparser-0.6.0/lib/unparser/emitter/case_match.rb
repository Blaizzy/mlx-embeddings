# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for case matches
    class CaseMatch < self

      handle :case_match

      children :target

      define_group :patterns, 1..-2

    private

      def else_branch
        children.last
      end

      def dispatch
        write('case ')
        visit(target)
        nl
        patterns.each(&method(:visit))
        nl unless buffer.fresh_line?
        emit_else_branch
        k_end
      end

      def emit_else_branch
        if else_branch
          write('else')
          emit_body(else_branch) unless n_empty_else?(else_branch)
          nl unless buffer.fresh_line?
        end
      end

    end # CaseMatch
  end # Emitter
end # Unparser
