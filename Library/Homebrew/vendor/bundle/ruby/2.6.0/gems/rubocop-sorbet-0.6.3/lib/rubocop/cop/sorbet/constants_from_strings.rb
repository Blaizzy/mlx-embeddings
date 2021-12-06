# frozen_string_literal: true

require "rubocop"

module RuboCop
  module Cop
    module Sorbet
      # This cop disallows the calls that are used to get constants fom Strings
      # such as +constantize+, +const_get+, and +constants+.
      #
      # The goal of this cop is to make the code easier to statically analyze,
      # more IDE-friendly, and more predictable. It leads to code that clearly
      # expresses which values the constant can have.
      #
      # @example
      #
      #   # bad
      #   class_name.constantize
      #
      #   # bad
      #   constants.detect { |c| c.name == "User" }
      #
      #   # bad
      #   const_get(class_name)
      #
      #   # good
      #   case class_name
      #   when "User"
      #     User
      #   else
      #     raise ArgumentError
      #   end
      #
      #   # good
      #   { "User" => User }.fetch(class_name)
      class ConstantsFromStrings < ::RuboCop::Cop::Cop
        def_node_matcher(:constant_from_string?, <<-PATTERN)
          (send _ {:constantize :constants :const_get} ...)
        PATTERN

        def on_send(node)
          return unless constant_from_string?(node)
          add_offense(
            node,
            location: :selector,
            message: "Don't use `#{node.method_name}`, it makes the code harder to understand, less editor-friendly, " \
              "and impossible to analyze. Replace `#{node.method_name}` with a case/when or a hash."
          )
        end
      end
    end
  end
end
