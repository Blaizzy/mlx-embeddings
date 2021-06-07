# frozen_string_literal: true

module RuboCop
  module Cop
    module Sorbet
      # This cop ensures that callback conditionals are bound to the right type
      # so that they are type checked properly.
      #
      # @example
      #
      #   # bad
      #   class Post < ApplicationRecord
      #     before_create :do_it, if: -> { should_do_it? }
      #
      #     def should_do_it?
      #       true
      #     end
      #   end
      #
      #   # good
      #   class Post < ApplicationRecord
      #     before_create :do_it, if: -> { T.bind(self, Post).should_do_it? }
      #
      #     def should_do_it?
      #       true
      #     end
      #   end
      class CallbackConditionalsBinding < RuboCop::Cop::Cop
        CALLBACKS = %i(
          validate
          validates
          validates_with
          before_validation
          around_validation

          before_create
          before_save
          before_destroy
          before_update

          after_create
          after_save
          after_destroy
          after_update
          after_touch
          after_initialize
          after_find

          around_create
          around_save
          around_destroy
          around_update

          before_commit

          after_commit
          after_create_commit
          after_destroy_commit
          after_rollback
          after_save_commit
          after_update_commit

          before_action
          prepend_before_action
          append_before_action

          around_action
          prepend_around_action
          append_around_action

          after_action
          prepend_after_action
          append_after_action
        ).freeze

        def autocorrect(node)
          lambda do |corrector|
            options = node.each_child_node.find(&:hash_type?)

            conditional = nil
            options.each_pair do |keyword, block|
              if keyword.value == :if || keyword.value == :unless
                conditional = block
                break
              end
            end

            _, _, block = conditional.child_nodes
            expected_class = node.parent_module_name

            bind = if block.begin_type?
              indentation = " " * block.child_nodes.first.loc.column
              "T.bind(self, #{expected_class})\n#{indentation}"
            elsif block.child_nodes.empty? && !block.ivar_type?
              "T.bind(self, #{expected_class})."
            else
              "T.bind(self, #{expected_class}); "
            end

            corrector.insert_before(block, bind)
          end
        end

        def on_send(node)
          return unless CALLBACKS.include?(node.method_name)

          options = node.each_child_node.find(&:hash_type?)
          return if options.nil?

          conditional = nil
          options.each_pair do |keyword, block|
            next unless keyword.sym_type?

            if keyword.value == :if || keyword.value == :unless
              conditional = block
              break
            end
          end

          return if conditional.nil? || conditional.child_nodes.empty?

          type, _, block = conditional.child_nodes
          return unless type.lambda_or_proc?

          expected_class = node.parent_module_name
          return if expected_class.nil?

          unless block.source.include?("T.bind(self, #{expected_class})")
            add_offense(
              node,
              message: "Callback conditionals should be bound to the right type. Use T.bind(self, #{expected_class})"
            )
          end
        end
      end
    end
  end
end
