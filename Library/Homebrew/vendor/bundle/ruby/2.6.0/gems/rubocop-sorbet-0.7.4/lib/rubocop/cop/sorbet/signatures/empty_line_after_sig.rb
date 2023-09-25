# frozen_string_literal: true

require_relative "signature_cop"

module RuboCop
  module Cop
    module Sorbet
      # Checks for blank lines after signatures.
      #
      # It also suggests an autocorrect
      #
      # @example
      #
      #   # bad
      #   sig { void }
      #
      #   def foo; end
      #
      #   # good
      #   sig { void }
      #   def foo; end
      #
      class EmptyLineAfterSig < SignatureCop
        include RangeHelp

        def on_signature(node)
          if (next_method(node).line - node.last_line) > 1
            location = source_range(processed_source.buffer, next_method(node).line - 1, 0)
            add_offense(node, location: location, message: "Extra empty line or comment detected")
          end
        end

        def autocorrect(node)
          ->(corrector) do
            offending_range = node.source_range.with(
              begin_pos: node.source_range.end_pos + 1,
              end_pos: processed_source.buffer.line_range(next_method(node).line).begin_pos,
            )
            corrector.remove(offending_range)
            clean_range = offending_range.source.split("\n").reject(&:empty?).join("\n")
            offending_line = processed_source.buffer.line_range(node.source_range.first_line)
            corrector.insert_before(offending_line, "#{clean_range}\n") unless clean_range.empty?
          end
        end

        private

        def next_method(node)
          processed_source.tokens.find do |t|
            t.line >= node.last_line &&
              (t.type == :kDEF || t.text.start_with?("attr_"))
          end
        end
      end
    end
  end
end
