# frozen_string_literal: true

module Unparser
  class Emitter
    # Range emitters
    class Range < self
      TOKENS = {
        irange: '..',
        erange: '...'
      }.freeze

      SYMBOLS = {
        erange: :tDOT3,
        irange: :tDOT2
      }.freeze

      def symbol_name
        true
      end

      handle(*TOKENS.keys)

      children :begin_node, :end_node

    private

      def dispatch
        visit(begin_node) if begin_node
        write(TOKENS.fetch(node.type))
        visit(end_node) if end_node
      end

    end # Range
  end # Emitter
end # Unparser
