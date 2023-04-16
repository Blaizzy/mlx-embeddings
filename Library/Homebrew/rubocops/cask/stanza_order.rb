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
          add_offenses(toplevel_stanzas)

          return unless (on_blocks = on_system_methods(toplevel_stanzas)).any?

          on_blocks.map(&:method_node).select(&:block_type?).each do |on_block|
            stanzas = inner_stanzas(on_block, processed_source.comments)
            add_offenses(stanzas, inner: true)
          end
        end

        private

        attr_reader :cask_block

        def_delegators :cask_block, :cask_node, :toplevel_stanzas,
                       :sorted_toplevel_stanzas, :sorted_inner_stanzas

        def add_offenses(stanzas, inner: false)
          sorted_stanzas = inner ? sorted_inner_stanzas(stanzas) : sorted_toplevel_stanzas
          offending_stanzas(stanzas, sorted_stanzas).each do |stanza|
            puts "offending stanza: #{stanza.stanza_name}"
            message = format(MESSAGE, stanza: stanza.stanza_name)
            add_offense(stanza.source_range_with_comments, message: message) do |corrector|
              correct_stanza_index = stanzas.index(stanza)
              correct_stanza = sorted_stanzas[correct_stanza_index]
              puts "correct stanza: #{correct_stanza.stanza_name}"
              corrector.replace(stanza.source_range_with_comments, correct_stanza.source_with_comments)
            end
          end
        end

        def offending_stanzas(stanzas, sorted_stanzas)
          stanza_pairs = stanzas.zip(sorted_stanzas)
          stanza_pairs.each_with_object([]) do |stanza_pair, offending_stanzas|
            stanza, sorted_stanza = *stanza_pair
            offending_stanzas << stanza if stanza != sorted_stanza
          end
        end
      end
    end
  end
end
