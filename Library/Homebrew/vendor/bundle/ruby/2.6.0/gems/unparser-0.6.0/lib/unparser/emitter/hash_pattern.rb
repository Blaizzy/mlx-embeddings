# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for hash patterns
    class HashPattern < self

      handle :hash_pattern

      def emit_const_pattern
        parentheses do
          emit_hash_body
        end
      end

    private

      def dispatch
        parentheses('{', '}') do
          emit_hash_body
        end
      end

      def emit_hash_body
        delimited(children, &method(:emit_member))
      end

      def emit_member(node)
        case node.type
        when :pair
          emit_pair(node)
        when :match_var
          emit_match_var(node)
        when :match_rest
          writer_with(MatchRest, node).emit_hash_pattern
        else
          visit(node)
        end
      end

      def emit_match_var(node)
        write_symbol_body(node.children.first)
        write(':')
      end

      def emit_pair(node)
        key, value = node.children

        if n_sym?(key)
          write_symbol_body(key.children.first)
        else
          visit(s(:dstr, *key))
        end

        write(':')

        ws

        visit(value)
      end

      def write_symbol_body(symbol)
        write(symbol.inspect[1..-1])
      end
    end # Pin
  end # Emitter
end # Unparser
