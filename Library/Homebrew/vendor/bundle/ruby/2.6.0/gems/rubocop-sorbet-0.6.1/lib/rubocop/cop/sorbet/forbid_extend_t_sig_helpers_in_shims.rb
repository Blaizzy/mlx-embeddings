# frozen_string_literal: true

module RuboCop
  module Cop
    module Sorbet
      # This cop ensures RBI shims do not include a call to extend T::Sig
      # or to extend T::Helpers
      #
      # @example
      #
      #   # bad
      #   module SomeModule
      #     extend T::Sig
      #     extend T::Helpers
      #
      #     sig { returns(String) }
      #     def foo; end
      #   end
      #
      #   # good
      #   module SomeModule
      #     sig { returns(String) }
      #     def foo; end
      #   end
      class ForbidExtendTSigHelpersInShims < RuboCop::Cop::Cop
        include RangeHelp

        MSG = 'Extending T::Sig or T::Helpers in a shim is unnecessary'
        RESTRICT_ON_SEND = [:extend]

        def_node_matcher :extend_t_sig?, <<~PATTERN
          (send nil? :extend (const (const nil? :T) :Sig))
        PATTERN

        def_node_matcher :extend_t_helpers?, <<~PATTERN
          (send nil? :extend (const (const nil? :T) :Helpers))
        PATTERN

        def autocorrect(node)
          -> (corrector) do
            corrector.remove(
              range_by_whole_lines(node.source_range, include_final_newline: true)
            )
          end
        end

        def on_send(node)
          add_offense(node) if extend_t_helpers?(node) || extend_t_sig?(node)
        end
      end
    end
  end
end
