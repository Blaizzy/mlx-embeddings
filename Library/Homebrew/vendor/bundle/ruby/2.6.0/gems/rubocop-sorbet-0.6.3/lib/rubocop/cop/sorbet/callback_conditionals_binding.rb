# frozen_string_literal: true

module RuboCop
  module Cop
    module Sorbet
      # This cop ensures that callback conditionals are bound to the right type
      # so that they are type checked properly.
      #
      # Auto-correction is unsafe because other libraries define similar style callbacks as Rails, but don't always need
      # binding to the attached class. Auto-correcting those usages can lead to false positives and auto-correction
      # introduces new typing errors.
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
      #     before_create :do_it, if: -> {
      #       T.bind(self, Post)
      #       should_do_it?
      #     }
      #
      #     def should_do_it?
      #       true
      #     end
      #   end
      class CallbackConditionalsBinding < RuboCop::Cop::Cop
        CALLBACKS = [
          :validate, :validates, :validates_with, :before_validation, :around_validation, :before_create,
          :before_save, :before_destroy, :before_update, :after_create, :after_save, :after_destroy,
          :after_update, :after_touch, :after_initialize, :after_find, :around_create, :around_save,
          :around_destroy, :around_update, :before_commit, :after_commit, :after_create_commit,
          :after_destroy_commit, :after_rollback, :after_save_commit, :after_update_commit,
          :before_action, :prepend_before_action, :append_before_action, :around_action,
          :prepend_around_action, :append_around_action, :after_action, :prepend_after_action,
          :append_after_action
        ].freeze

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

            # Find the class node and check if it includes a namespace on the
            # same line e.g.: Namespace::Class, which will require the fully
            # qualified name

            klass = node.ancestors.find(&:class_type?)

            expected_class = if klass.children.first.children.first.nil?
              node.parent_module_name.split("::").last
            else
              klass.identifier.source
            end

            do_end_lambda = conditional.source.include?("do") && conditional.source.include?("end")

            unless do_end_lambda
              # We are converting a one line lambda into a multiline
              # Remove the space after the `{`
              if /{\s/.match?(conditional.source)
                corrector.remove_preceding(block, 1)
              end

              # Remove the last space and `}` and re-add it with a line break
              # and the correct indentation
              base_indentation = " " * node.loc.column
              chars_to_remove = /\s}/.match?(conditional.source) ? 2 : 1
              corrector.remove_trailing(conditional, chars_to_remove)
              corrector.insert_after(block, "\n#{base_indentation}}")
            end

            # Add the T.bind
            indentation = " " * (node.loc.column + 2)
            line_start = do_end_lambda ? "" : "\n#{indentation}"
            bind = "#{line_start}T.bind(self, #{expected_class})\n#{indentation}"

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

          return if conditional.nil? || conditional.array_type? || conditional.child_nodes.empty?

          return unless conditional.arguments.empty?

          type, _, block = conditional.child_nodes
          return unless type.lambda_or_proc? || type.block_literal?

          klass = node.ancestors.find(&:class_type?)

          expected_class = if klass&.children&.first&.children&.first.nil?
            node.parent_module_name&.split("::")&.last
          else
            klass.identifier.source
          end

          return if expected_class.nil?

          unless block.source.include?("T.bind(self")
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
