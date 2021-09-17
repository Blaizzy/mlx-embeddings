# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for in pattern nodes
    class MatchVar < self

      handle :match_var

      children :name

    private

      def dispatch
        write(name.to_s)
      end
    end # MatchVar
  end # Emitter
end # Unparser
