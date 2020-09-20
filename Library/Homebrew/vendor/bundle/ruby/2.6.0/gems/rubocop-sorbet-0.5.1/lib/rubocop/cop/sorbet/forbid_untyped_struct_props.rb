# encoding: utf-8
# frozen_string_literal: true

require 'rubocop'

module RuboCop
  module Cop
    module Sorbet
      # This cop disallows use of `T.untyped` or `T.nilable(T.untyped)`
      # as a prop type for `T::Struct`.
      #
      # @example
      #
      #   # bad
      #   class SomeClass
      #     const :foo, T.untyped
      #     prop :bar, T.nilable(T.untyped)
      #   end
      #
      #   # good
      #   class SomeClass
      #     const :foo, Integer
      #     prop :bar, T.nilable(String)
      #   end
      class ForbidUntypedStructProps < RuboCop::Cop::Cop
        MSG = 'Struct props cannot be T.untyped'

        def_node_matcher :t_struct, <<~PATTERN
          (const (const nil? :T) :Struct)
        PATTERN

        def_node_matcher :t_untyped, <<~PATTERN
          (send (const nil? :T) :untyped)
        PATTERN

        def_node_matcher :t_nilable_untyped, <<~PATTERN
          (send (const nil? :T) :nilable {#t_untyped #t_nilable_untyped})
        PATTERN

        def_node_matcher :subclass_of_t_struct?, <<~PATTERN
          (class (const ...) #t_struct ...)
        PATTERN

        def_node_search :untyped_props, <<~PATTERN
          (send nil? {:prop :const} _ {#t_untyped #t_nilable_untyped} ...)
        PATTERN

        def on_class(node)
          return unless subclass_of_t_struct?(node)

          untyped_props(node).each do |untyped_prop|
            add_offense(untyped_prop.child_nodes[1])
          end
        end
      end
    end
  end
end
