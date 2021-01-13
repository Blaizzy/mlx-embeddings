# typed: strict
# frozen_string_literal: true

module RuboCop
  module Cop
    module Style
      # This cop checks that `unless` is not used with multiple conditions.
      #
      # @api private
      class UnlessMultipleConditions < Base
        extend T::Sig
        extend AutoCorrector

        MSG = "Avoid using `unless` with multiple conditions."

        sig { params(node: RuboCop::AST::IfNode).void }
        def on_if(node)
          return if !node.unless? || (!node.condition.and_type? && !node.condition.or_type?)

          add_offense(node.condition.source_range.with(begin_pos: node.loc.keyword.begin_pos)) do |corrector|
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
