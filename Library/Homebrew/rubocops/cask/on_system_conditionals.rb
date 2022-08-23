# typed: true
# frozen_string_literal: true

require "forwardable"
require "rubocops/shared/on_system_conditionals_helper"

module RuboCop
  module Cop
    module Cask
      # This cop makes sure that OS conditionals are consistent.
      #
      # @example
      #   # bad
      #   cask 'foo' do
      #     if MacOS.version == :high_sierra
      #       sha256 "..."
      #     end
      #   end
      #
      #   # good
      #   cask 'foo' do
      #     on_high_sierra do
      #       sha256 "..."
      #     end
      #   end
      class OnSystemConditionals < Base
        extend Forwardable
        extend AutoCorrector
        include OnSystemConditionalsHelper
        include CaskHelp

        FLIGHT_STANZA_NAMES = [:preflight, :postflight, :uninstall_preflight, :uninstall_postflight].freeze

        def on_cask(cask_block)
          @cask_block = cask_block

          toplevel_stanzas.each do |stanza|
            next unless FLIGHT_STANZA_NAMES.include? stanza.stanza_name

            audit_on_system_blocks(stanza.stanza_node, stanza.stanza_name)
          end
        end

        private

        attr_reader :cask_block

        def_delegators :cask_block, :toplevel_stanzas, :cask_body
      end
    end
  end
end
