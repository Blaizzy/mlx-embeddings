# frozen_string_literal: true

require 'rubocop'

module RuboCop
  module Cop
    module Sorbet
      # This cop disallows using `.override(allow_incompatible: true)`.
      # Using `allow_incompatible` suggests a violation of the Liskov
      # Substitution Principle, meaning that a subclass is not a valid
      # subtype of it's superclass. This Cop prevents these design smells
      # from occurring.
      #
      # @example
      #
      #   # bad
      #   sig.override(allow_incompatible: true)
      #
      #   # good
      #   sig.override
      class AllowIncompatibleOverride < RuboCop::Cop::Cop
        def_node_search(:sig?, <<-PATTERN)
          (
            send
            nil?
            :sig
             ...
          )
        PATTERN

        def not_nil?(node)
          !node.nil?
        end

        def_node_search(:allow_incompatible?, <<-PATTERN)
          (pair (sym :allow_incompatible) (true))
        PATTERN

        def_node_matcher(:allow_incompatible_override?, <<-PATTERN)
          (
            send
            [#not_nil? #sig?]
            :override
            [#not_nil? #allow_incompatible?]
          )
        PATTERN

        def on_send(node)
          return unless allow_incompatible_override?(node)
          add_offense(
            node.children[2],
            message: 'Usage of `allow_incompatible` suggests a violation of the Liskov Substitution Principle. '\
            'Instead, strive to write interfaces which respect subtyping principles and remove `allow_incompatible`',
          )
        end
      end
    end
  end
end
