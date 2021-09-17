# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter control flow modifiers
    class FlowModifier < self
      MAP = {
        return: 'return',
        next:   'next',
        break:  'break'
      }.freeze

      private_constant(*constants(false))

      handle(*MAP.keys)

      def emit_heredoc_reminders
        children.each do |node|
          emitter(node).emit_heredoc_reminders
        end
      end

    private

      def dispatch
        write(MAP.fetch(node.type))

        if children.one? && n_if?(children.first)
          ws
          emitter(children.first).emit_ternary
        else
          emit_arguments unless children.empty?
        end
      end

      def emit_arguments
        ws
        delimited(children)
      end
    end # Return
  end # Emitter
end # Unparser
