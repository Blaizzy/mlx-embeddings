# typed: strict
# frozen_string_literal: true

module RuboCop
  module Cop
    module Style
      # This cop checks that `unless` is not used with multiple conditions.
      #
      # @api private
      class UnlessMultipleConditions < Cop
        extend T::Sig

        MSG = "Avoid using `unless` with multiple conditions."

        sig { params(node: RuboCop::AST::IfNode).void }
        def on_if(node)
          return if !node.unless? || (!node.condition.and_type? && !node.condition.or_type?)

          add_offense(node, location: node.condition.source_range.with(begin_pos: node.loc.keyword.begin_pos))
        end

        sig { params(node: RuboCop::AST::IfNode).returns(T.proc.params(arg0: RuboCop::Cop::Corrector).void) }
        def autocorrect(node)
          lambda do |corrector|
            corrector.replace(node.loc.keyword, "if")
            corrector.replace(node.condition.loc.operator, node.condition.inverse_operator)
            [node.condition.lhs, node.condition.rhs].each do |subcondition|
              if !subcondition.source.start_with?("(") || !subcondition.source.end_with?(")")
                corrector.insert_before(subcondition, "(")
                corrector.insert_after(subcondition, ")")
              end
              corrector.insert_before(subcondition, "!")
            end
          end
        end
      end
    end
  end
end
