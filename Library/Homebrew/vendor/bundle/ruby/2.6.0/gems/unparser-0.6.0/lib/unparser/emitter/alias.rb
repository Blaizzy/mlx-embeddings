# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for alias nodes
    class Alias < self

      handle :alias

      children :target, :source

    private

      def dispatch
        write('alias ')
        visit(target)
        ws
        visit(source)
      end

    end # Alias
  end # Emitter
end # Unparser
