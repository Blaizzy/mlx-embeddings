# frozen_string_literal: true

module Unparser
  class Emitter
    # Dynamic string emitter
    class DStr < self

      handle :dstr

      def emit_heredoc_reminders
        writer_with(Writer::DynamicString, node).emit_heredoc_reminder
      end

    private

      def dispatch
        writer_with(Writer::DynamicString, node).dispatch
      end

    end # DStr
  end # Emitter
end # Unparser
