# frozen_string_literal: true

require "rubocop"

module RuboCop
  module Cop
    module Sorbet
      # This cop disallows using `T.untyped` anywhere.
      #
      # @example
      #
      #   # bad
      #   sig { params(my_argument: T.untyped).void }
      #   def foo(my_argument); end
      #
      #   # good
      #   sig { params(my_argument: String).void }
      #   def foo(my_argument); end
      #
      class ForbidTUntyped < RuboCop::Cop::Cop
        def_node_matcher(:t_untyped?, "(send (const nil? :T) :untyped)")

        def on_send(node)
          add_offense(node, message: "Do not use `T.untyped`.") if t_untyped?(node)
        end
      end
    end
  end
end
