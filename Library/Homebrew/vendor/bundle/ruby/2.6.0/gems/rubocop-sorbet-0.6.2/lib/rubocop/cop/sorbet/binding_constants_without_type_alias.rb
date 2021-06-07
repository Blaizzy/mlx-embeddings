# frozen_string_literal: true

require 'rubocop'

module RuboCop
  module Cop
    module Sorbet
      # This cop disallows binding the return value of `T.any`, `T.all`, `T.enum`
      # to a constant directly. To bind the value, one must use `T.type_alias`.
      #
      # @example
      #
      #   # bad
      #   FooOrBar = T.any(Foo, Bar)
      #
      #   # good
      #   FooOrBar = T.type_alias { T.any(Foo, Bar) }
      class BindingConstantWithoutTypeAlias < RuboCop::Cop::Cop
        def_node_matcher(:binding_unaliased_type?, <<-PATTERN)
          (casgn _ _ [#not_nil? #not_t_let? #not_dynamic_type_creation_with_block? #not_generic_parameter_decl? #method_needing_aliasing_on_t?])
        PATTERN

        def_node_matcher(:using_type_alias?, <<-PATTERN)
          (block
            (send
              (const nil? :T) :type_alias)
              _
              _
          )
        PATTERN

        def_node_matcher(:using_deprecated_type_alias_syntax?, <<-PATTERN)
          (
            send
            (const nil? :T)
            :type_alias
            _
          )
        PATTERN

        def_node_matcher(:t_let?, <<-PATTERN)
          (
            send
            (const nil? :T)
            :let
            _
            _
          )
        PATTERN

        def_node_matcher(:dynamic_type_creation_with_block?, <<-PATTERN)
          (block
            (send
              const :new ...)
              _
              _
          )
        PATTERN

        def_node_matcher(:generic_parameter_decl?, <<-PATTERN)
          (
            send nil? {:type_template :type_member} ...
          )
        PATTERN

        def_node_search(:method_needing_aliasing_on_t?, <<-PATTERN)
          (
            send
            (const nil? :T)
            {:any :all :noreturn :class_of :untyped :nilable :self_type :enum :proc}
             ...
          )
        PATTERN

        def not_t_let?(node)
          !t_let?(node)
        end

        def not_dynamic_type_creation_with_block?(node)
          !dynamic_type_creation_with_block?(node)
        end

        def not_generic_parameter_decl?(node)
          !generic_parameter_decl?(node)
        end

        def not_nil?(node)
          !node.nil?
        end

        def on_casgn(node)
          return unless binding_unaliased_type?(node) && !using_type_alias?(node.children[2])
          if using_deprecated_type_alias_syntax?(node.children[2])
            add_offense(
              node.children[2],
              message: "It looks like you're using the old `T.type_alias` syntax. " \
              '`T.type_alias` now expects a block.' \
              'Run Sorbet with the options "--autocorrect --error-white-list=5043" ' \
              'to automatically upgrade to the new syntax.'
            )
            return
          end
          add_offense(
            node.children[2],
            message: "It looks like you're trying to bind a type to a constant. " \
            'To do this, you must alias the type using `T.type_alias`.'
          )
        end

        def autocorrect(node)
          lambda do |corrector|
            corrector.replace(
              node.source_range,
              "T.type_alias { #{node.source} }"
            )
          end
        end
      end
    end
  end
end
