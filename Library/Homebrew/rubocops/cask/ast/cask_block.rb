# typed: true
# frozen_string_literal: true

require "forwardable"

module RuboCop
  module Cask
    module AST
      # This class wraps the AST block node that represents the entire cask
      # definition. It includes various helper methods to aid cops in their
      # analysis.
      class CaskBlock
        extend Forwardable

        def initialize(block_node, comments)
          @block_node = block_node
          @comments = comments
        end

        attr_reader :block_node, :comments

        alias cask_node block_node

        def_delegator :cask_node, :block_body, :cask_body

        def header
          @header ||= CaskHeader.new(cask_node.method_node)
        end

        def stanzas
          return [] unless cask_body

          @stanzas ||= cask_body.each_node
                                .select(&:stanza?)
                                .map { |node| Stanza.new(node, cask_node) }
        end

        def toplevel_stanzas
          # If a `cask` block only contains one stanza, it is that stanza's direct parent,
          # otherwise stanzas are grouped in a block and `cask` is that block's parent.
          is_toplevel_stanza = if cask_body.begin_block?
            ->(stanza) { stanza.parent_node.parent.cask_block? }
          else
            ->(stanza) { stanza.parent_node.cask_block? }
          end

          @toplevel_stanzas ||= stanzas.select(&is_toplevel_stanza)
        end

        def sorted_toplevel_stanzas
          @sorted_toplevel_stanzas ||= sort_stanzas(toplevel_stanzas)
        end

        private

        def sort_stanzas(stanzas)
          stanzas.sort do |s1, s2|
            i1 = stanza_order_index(s1)
            i2 = stanza_order_index(s2)
            if i1 == i2 || i1.blank? || i2.blank?
              i1 = stanzas.index(s1)
              i2 = stanzas.index(s2)
            end
            i1 - i2
          end
        end

        def stanza_order_index(stanza)
          Constants::STANZA_ORDER.index(stanza.stanza_name)
        end
      end
    end
  end
end
