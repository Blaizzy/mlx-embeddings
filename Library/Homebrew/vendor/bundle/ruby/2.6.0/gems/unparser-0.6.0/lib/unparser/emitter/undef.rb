# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for undef nodes
    class Undef < self
      handle :undef

    private

      def dispatch
        write('undef ')
        delimited(children)
      end

    end # Undef
  end # Emitter
end # Unparser
