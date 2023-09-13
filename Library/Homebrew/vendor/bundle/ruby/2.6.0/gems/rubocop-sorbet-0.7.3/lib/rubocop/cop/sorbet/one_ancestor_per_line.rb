# encoding: utf-8
# frozen_string_literal: true

require "rubocop"

module RuboCop
  module Cop
    module Sorbet
      # Ensures one ancestor per requires_ancestor line
      # rather than chaining them as a comma-separated list.
      #
      # @example
      #
      #   # bad
      #   module SomeModule
      #     requires_ancestor Kernel, Minitest::Assertions
      #   end
      #
      #   # good
      #   module SomeModule
      #     requires_ancestor Kernel
      #     requires_ancestor Minitest::Assertions
      #   end
      class OneAncestorPerLine < RuboCop::Cop::Cop # rubocop:todo InternalAffairs/InheritDeprecatedCopClass
        MSG = "Cannot require more than one ancestor per line"

        # @!method requires_ancestors(node)
        def_node_search :requires_ancestors, <<~PATTERN
          (send nil? :requires_ancestor ...)
        PATTERN

        # @!method more_than_one_ancestor(node)
        def_node_matcher :more_than_one_ancestor, <<~PATTERN
          (send nil? :requires_ancestor const const+)
        PATTERN

        # @!method abstract?(node)
        def_node_search :abstract?, <<~PATTERN
          (send nil? :abstract!)
        PATTERN

        def on_module(node)
          return unless node.body
          return unless requires_ancestors(node)

          process_node(node)
        end

        def on_class(node)
          return unless abstract?(node)
          return unless requires_ancestors(node)

          process_node(node)
        end

        def autocorrect(node)
          ->(corrector) do
            ra_call = node.parent
            split_ra_calls = ra_call.source.gsub(/,\s+/, new_ra_line(ra_call.loc.column))
            corrector.replace(ra_call, split_ra_calls)
          end
        end

        private

        def process_node(node)
          requires_ancestors(node).each do |ra|
            add_offense(ra.child_nodes[1]) if more_than_one_ancestor(ra)
          end
        end

        def new_ra_line(indent_count)
          indents = " " * indent_count
          indented_ra_call = "#{indents}requires_ancestor "
          "\n#{indented_ra_call}"
        end
      end
    end
  end
end
