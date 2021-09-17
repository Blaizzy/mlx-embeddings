# frozen_string_literal: true

module Unparser
  class Emitter
    class Kwargs < self
      handle :kwargs

      def dispatch
        delimited(children)
      end
    end # Kwargs
  end # Emitter
end # Unparser
