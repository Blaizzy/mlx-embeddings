# frozen_string_literal: true

module Unparser
  module NodeDetails
    class Send
      include NodeDetails

      ASSIGN_SUFFIX    = '='.freeze
      NON_ASSIGN_RANGE = (0..-2).freeze

      private_constant(*constants(false))

      children :receiver, :selector

      public :receiver, :selector

      def selector_binary_operator?
        BINARY_OPERATORS.include?(selector)
      end

      def binary_syntax_allowed?
        selector_binary_operator? && arguments.one? && !n_splat?(arguments.first)
      end

      def selector_unary_operator?
        UNARY_OPERATORS.include?(selector)
      end

      def assignment_operator?
        assignment? && !selector_binary_operator? && !selector_unary_operator?
      end

      def arguments?
        arguments.any?
      end

      def non_assignment_selector
        if assignment?
          string_selector[NON_ASSIGN_RANGE]
        else
          string_selector
        end
      end

      def assignment?
        string_selector[-1].eql?(ASSIGN_SUFFIX)
      end
      memoize :assignment?

      def arguments
        children[2..-1]
      end
      memoize :arguments

      def string_selector
        selector.to_s
      end
      memoize :string_selector

    end # Send
  end # NodeDetails
end # Unparser
