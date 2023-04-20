# typed: true
# frozen_string_literal: true

require "forwardable"

module RuboCop
  module Cop
    module Cask
      # This cop checks that a cask's stanzas are ordered correctly, including nested within `on_*` blocks.
      # @see https://docs.brew.sh/Cask-Cookbook#stanza-order
      class StanzaOrder < Base
        extend Forwardable
        extend AutoCorrector
        include CaskHelp

        MESSAGE = "`%<stanza>s` stanza out of order"

        def on_cask(cask_block)
          @cask_block = cask_block

          # Find all the stanzas that are direct children of the cask block or one of its `on_*` blocks.
          puts "toplevel_stanzas: #{toplevel_stanzas.map(&:stanza_name).inspect}"
          outer_and_inner_stanzas = toplevel_stanzas + toplevel_stanzas.map do |stanza|
            return stanza unless stanza.method_node&.block_type?

            inner_stanzas(stanza.method_node, stanza.comments)
          end

          puts "outer_and_inner_stanzas: #{outer_and_inner_stanzas.flatten.map(&:stanza_name).inspect}"
          add_offenses(outer_and_inner_stanzas.flatten)
        end

        private

        attr_reader :cask_block

        def_delegators :cask_block, :cask_node, :toplevel_stanzas

        def add_offenses(outer_and_inner_stanzas)
          outer_and_inner_stanzas.each_cons(2) do |stanza1, stanza2|
            next if stanza_order_index(stanza1.stanza_name) < stanza_order_index(stanza2.stanza_name)

            puts "#{stanza2.stanza_name} should come before #{stanza1.stanza_name}"
            add_offense(stanza1.method_node, message: format(MESSAGE, stanza: stanza1.stanza_name)) do |corrector|
              # TODO: Move the stanza to the correct location.
            end
          end
        end

        def stanza_order_index(stanza_name)
          RuboCop::Cask::Constants::STANZA_ORDER.index(stanza_name)
        end
      end
    end
  end
end
