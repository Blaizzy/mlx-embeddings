# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for send
    class Send < self
      handle :csend, :send

      def emit_mlhs
        writer.emit_mlhs
      end

      def emit_heredoc_reminders
        writer.emit_heredoc_reminders
      end

    private

      def dispatch
        writer.dispatch
      end

      def writer
        writer_with(Writer::Send, node)
      end
      memoize :writer
    end # Send
  end # Emitter
end # Unparser
