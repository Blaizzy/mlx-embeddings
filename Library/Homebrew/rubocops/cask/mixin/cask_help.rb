# typed: true
# frozen_string_literal: true

module RuboCop
  module Cop
    module Cask
      # Common functionality for cops checking casks.
      module CaskHelp
        extend T::Helpers

        abstract!

        sig { abstract.params(cask_block: RuboCop::Cask::AST::CaskBlock).void }
        def on_cask(cask_block); end

        def on_block(block_node)
          super if defined? super
          return unless respond_to?(:on_cask)
          return unless block_node.cask_block?

          comments = processed_source.comments
          cask_block = RuboCop::Cask::AST::CaskBlock.new(block_node, comments)
          on_cask(cask_block)
        end
      end
    end
  end
end
