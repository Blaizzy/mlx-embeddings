# frozen_string_literal: true

module Unparser
  class Emitter

    # Emitter for begin nodes
    class Begin < self
      handle :begin
      children :body

      def emit_heredoc_reminders
        children.each do |child|
          emitter(child).emit_heredoc_reminders
        end
      end

    private

      def dispatch
        parentheses do
          delimited(children, '; ')
        end
      end
    end # Begin
  end # Emitter
end # Unparser
