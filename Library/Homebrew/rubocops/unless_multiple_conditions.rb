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
          add_offense(node) if node.unless? && (node.condition.and_type? || node.condition.or_type?)
        end
      end
    end
  end
end
