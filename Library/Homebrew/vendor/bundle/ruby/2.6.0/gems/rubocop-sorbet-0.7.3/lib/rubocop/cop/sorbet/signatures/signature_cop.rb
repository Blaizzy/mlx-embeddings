# frozen_string_literal: true

require "rubocop"

module RuboCop
  module Cop
    module Sorbet
      # Abstract cop specific to Sorbet signatures
      #
      # You can subclass it to use the `on_signature` trigger and the `signature?` node matcher.
      class SignatureCop < RuboCop::Cop::Cop # rubocop:todo InternalAffairs/InheritDeprecatedCopClass
        @registry = Cop.registry # So we can properly subclass this cop

        # @!method signature?(node)
        def_node_matcher(:signature?, <<~PATTERN)
          (block (send
            {nil? #with_runtime? #without_runtime?}
            :sig
            (sym :final)?
          ) (args) ...)
        PATTERN

        # @!method with_runtime?(node)
        def_node_matcher(:with_runtime?, <<~PATTERN)
          (const (const nil? :T) :Sig)
        PATTERN

        # @!method without_runtime?(node)
        def_node_matcher(:without_runtime?, <<~PATTERN)
          (const (const (const nil? :T) :Sig) :WithoutRuntime)
        PATTERN

        def on_block(node)
          on_signature(node) if signature?(node)
        end

        alias_method :on_numblock, :on_block

        def on_signature(_)
          # To be defined in subclasses
        end
      end
    end
  end
end
