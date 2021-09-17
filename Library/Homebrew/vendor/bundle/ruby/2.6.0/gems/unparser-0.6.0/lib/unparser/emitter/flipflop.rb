# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for flip flops
    class FlipFlop < self
      MAP = {
        iflipflop: '..',
        eflipflop: '...'
      }.freeze

      SYMBOLS = {
        eflipflop: :tDOT3,
        iflipflop: :tDOT2
      }.freeze

      def symbol_name
        true
      end

      handle(*MAP.keys)

      children :left, :right

    private

      def dispatch
        visit(left)
        write(MAP.fetch(node.type))
        visit(right)
      end
    end # FlipFLop
  end # Emitter
end # Unparser
