# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for Hash literals
    class Hash < self
      handle :hash

      def emit_last_argument_hash
        if children.empty?
          write('{}')
        else
          emit_hash_body
        end
      end

      def emit_heredoc_reminders
        children.each(&method(:emit_heredoc_reminder_member))
      end

    private

      def dispatch
        if children.empty?
          write('{}')
        else
          parentheses('{', '}') do
            write(' ')
            emit_hash_body
            write(' ')
          end
        end
      end

      def emit_heredoc_reminder_member(node)
        emitter(node.children.last).emit_heredoc_reminders
      end

      def emit_hash_body
        delimited(children)
      end
    end # Hash
  end # Emitter
end # Unparser
