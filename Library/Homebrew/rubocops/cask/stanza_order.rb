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
          stanzas = [toplevel_stanzas]

          puts "before on blocks: #{stanzas.first.map(&:stanza_name)}"
          if (on_blocks = on_system_methods(stanzas.first)).any?
            on_blocks.map(&:method_node).select(&:block_type?).each do |on_block|
              stanzas.push(inner_stanzas(on_block, processed_source.comments))
            end
          end

          puts "after on blocks: #{stanzas.last.map(&:method_node).select(&:send_type?).map(&:method_name) }" if on_blocks
          add_offenses(stanzas)
        end

        private

        attr_reader :cask_block

        def_delegators :cask_block, :cask_node, :toplevel_stanzas,
                       :sorted_toplevel_stanzas

        def add_offenses(outer_and_inner_stanzas)
          outer_and_inner_stanzas.map do |stanza_types|
            offending_stanzas(stanza_types, sorted_toplevel_stanzas).flatten.compact.each do |stanza|
              name = stanza.respond_to?(:method_name) ? stanza.method_name : stanza.stanza_name
              message = format(MESSAGE, stanza: name)
              add_offense(stanza.source_range_with_comments, message: message) do |corrector|
                correct_stanza_index = outer_and_inner_stanzas.flatten.index(stanza)
                correct_stanza = sorted_toplevel_stanzas[correct_stanza_index]
                corrector.replace(stanza&.source_range_with_comments, correct_stanza&.source_with_comments)
              end
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
