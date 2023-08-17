# encoding: utf-8
# frozen_string_literal: true

require "rubocop"

module RuboCop
  module Cop
    module Sorbet
      # Correct `send` expressions in include statements by constant literals.
      #
      # Sorbet, the static checker, is not (yet) able to support constructs on the
      # following form:
      #
      # ```ruby
      # class MyClass
      #   include send_expr
      # end
      # ```
      #
      # Multiple occurences of this can be found in Shopify's code base like:
      #
      # ```ruby
      # include Rails.application.routes.url_helpers
      # ```
      # or
      # ```ruby
      # include Polaris::Engine.helpers
      # ```
      class ForbidIncludeConstLiteral < RuboCop::Cop::Cop # rubocop:todo InternalAffairs/InheritDeprecatedCopClass
        MSG = "Includes must only contain constant literals"

        attr_accessor :used_names

        # @!method not_lit_const_include?(node)
        def_node_matcher :not_lit_const_include?, <<-PATTERN
          (send nil? {:include :extend :prepend}
            $_
          )
        PATTERN

        def initialize(*)
          super
          self.used_names = Set.new
        end

        def on_send(node)
          return unless not_lit_const_include?(node) do |send_argument|
            ![:const, :self].include?(send_argument.type)
          end

          parent = node.parent
          return unless parent

          parent = parent.parent if [:begin, :block].include?(parent.type)
          return unless [:module, :class, :sclass].include?(parent.type)

          add_offense(node)
        end

        def autocorrect(node)
          lambda do |corrector|
            corrector.replace(
              node,
              "T.unsafe(self).#{node.source}",
            )
          end
        end
      end
    end
  end
end
