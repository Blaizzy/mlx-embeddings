# frozen_string_literal: true

require "rubocop"

module RuboCop
  module Cop
    module Sorbet
      # This cop ensures all constants used as `T.type_alias` are using CamelCase.
      #
      # @example
      #
      #   # bad
      #   FOO_OR_BAR = T.type_alias { T.any(Foo, Bar) }
      #
      #   # good
      #   FooOrBar = T.type_alias { T.any(Foo, Bar) }
      class TypeAliasName < RuboCop::Cop::Cop
        MSG = "Type alias constant name should be in CamelCase"

        def_node_matcher(:casgn_type_alias?, <<-PATTERN)
          (casgn
            _
            _
            (block
              (send
                (const nil? :T) :type_alias)
                _
                _
            ))
        PATTERN

        def on_casgn(node)
          return unless casgn_type_alias?(node)

          name = node.children[1]

          # From https://github.com/rubocop/rubocop/blob/master/lib/rubocop/cop/naming/class_and_module_camel_case.rb
          return unless /_/.match?(name)

          add_offense(node)
        end
      end
    end
  end
end
