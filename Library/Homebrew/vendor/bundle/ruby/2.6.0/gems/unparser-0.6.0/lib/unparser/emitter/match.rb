# frozen_string_literal: true

module Unparser
  class Emitter

    # Base class for special match node emitters
    class Match < self
      # Emitter for match with local variable assignment
      class Lvasgn < self
        handle :match_with_lvasgn

        children :regexp, :lvasgn

      private

        def dispatch
          visit(regexp)
          write(' =~ ')
          visit(lvasgn)
        end

      end # Lvasgn

      # Emitter for match current line
      class CurrentLine < self
        handle :match_current_line

        children :regexp

      private

        def dispatch
          visit(regexp)
        end

      end # CurrentLine

    end # Match
  end # Emitter
end # Unparser
