# frozen_string_literal: true

module Unparser
  module Writer
    class Binary
      include Writer, Adamantium

      children :left, :right

      OPERATOR_TOKENS =
        {
          and: '&&',
          or:  '||'
        }.freeze

      KEYWORD_TOKENS =
        {
          and: 'and',
          or:  'or'
        }.freeze

      KEYWORD_SYMBOLS =
        {
          and: :kAND,
          or:  :kOR
        }.freeze

      OPERATOR_SYMBOLS =
        {
          and: :tANDOP,
          or:  :tOROP
        }.freeze

      MAP =
        {
          kAND:   'and',
          kOR:    'or',
          tOROP:  '||',
          tANDOP: '&&'
        }.freeze

      NEED_KEYWORD = %i[return break next].freeze

      private_constant(*constants(false))

      def emit_operator
        emit_with(OPERATOR_TOKENS)
      end

      def symbol_name
        true
      end

      def dispatch
        left_emitter.write_to_buffer
        write(' ', MAP.fetch(effective_symbol), ' ')
        visit(right)
      end

    private

      def effective_symbol
        if NEED_KEYWORD.include?(right.type) || NEED_KEYWORD.include?(left.type)
          return keyword_symbol
        end

        unless left_emitter.symbol_name
          return operator_symbol
        end

        keyword_symbol
      end

      def emit_with(map)
        visit(left)
        write(' ', map.fetch(node.type), ' ')
        visit(right)
      end

      def keyword_symbol
        KEYWORD_SYMBOLS.fetch(node.type)
      end

      def operator_symbol
        OPERATOR_SYMBOLS.fetch(node.type)
      end

      def left_emitter
        emitter(left)
      end
      memoize :left_emitter

      def right_emitter
        emitter(right)
      end
      memoize :right_emitter
    end # Binary
  end # Writer
end # Unparser
