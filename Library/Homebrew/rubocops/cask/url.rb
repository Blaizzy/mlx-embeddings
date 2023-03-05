# typed: true
# frozen_string_literal: true

module RuboCop
  module Cop
    module Cask
      # This cop checks that a cask's `url` stanza is formatted correctly.
      #
      # @example
      #   # bad
      #   url "https://example.com/foo.dmg",
      #     verified: "https://example.com"
      #
      #
      #   # good
      #   url "https://example.com/foo.dmg",
      #     verified: "example.com"
      #
      class Url < Base
        extend AutoCorrector
        extend Forwardable
        include OnUrlStanza

        def on_url_stanza(stanza)
          return if stanza.stanza_node.block_type?

          hash_node = stanza.stanza_node.last_argument
          return unless hash_node.hash_type?

          hash_node.each_pair do |key_node, value_node|
            next unless key_node.source == "verified"
            next unless value_node.str_type?
            next unless value_node.source.start_with?(%r{^"https?://})

            add_offense(
              value_node.source_range,
              message: "Verified URL parameter value should not start with https:// or http://.",
            ) do |corrector|
              corrector.replace(value_node.source_range, value_node.source.gsub(%r{^"https?://}, "\""))
            end
          end
        end
      end
    end
  end
end
