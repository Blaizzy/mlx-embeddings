# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for key value pairs in hash literals or kwargs
    class Pair < self
      BAREWORD = /\A[A-Za-z_][A-Za-z_0-9]*[?!]?\z/.freeze

      private_constant(*constants(false))

      handle :pair

      children :key, :value

    private

      def dispatch
        if colon?(key)
          write(key.children.first.to_s, ': ')
        else
          visit(key)
          write(' => ')
        end

        visit(value)
      end

      def colon?(key)
        n_sym?(key) && BAREWORD.match?(key.children.first)
      end
    end
  end
end
