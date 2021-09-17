# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for lambda nodes
    class Lambda < self
      handle :lambda

    private

      def dispatch
        write('->')
      end

    end # Lambda
  end # Emitter
end # Unparser
