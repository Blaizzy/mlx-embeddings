# frozen_string_literal: true

require 'rubocop'

module RuboCop
  module Cop
    module Sorbet
      # Abstract cop specific to Sorbet signatures
      #
      # You can subclass it to use the `on_signature` trigger and the `signature?` node matcher.
      class SignatureCop < RuboCop::Cop::Cop
        @registry = Cop.registry # So we can properly subclass this cop

        def_node_matcher(:signature?, <<~PATTERN)
          (block (send nil? :sig) (args) ...)
        PATTERN

        def on_block(node)
          on_signature(node) if signature?(node)
        end

        def on_signature(_)
          # To be defined in subclasses
        end
      end
    end
  end
end
