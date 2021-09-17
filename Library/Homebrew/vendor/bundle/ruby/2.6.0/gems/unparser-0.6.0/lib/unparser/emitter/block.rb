# frozen_string_literal: true

module Unparser
  class Emitter

    # Block emitter
    class Block < self
      handle :block, :numblock

      children :target, :arguments, :body

    private

      def dispatch
        emit_target
        ws
        write_open
        target_writer.emit_heredoc_reminders if n_send?(target)
        emit_block_arguments unless n_lambda?(target)
        emit_optional_body_ensure_rescue(body)
        write_close
      end

      def need_do?
        body && (n_rescue?(body) || n_ensure?(body))
      end

      def write_open
        if need_do?
          write('do')
        else
          write('{')
        end
      end

      def write_close
        if need_do?
          k_end
        else
          write('}')
        end
      end

      def target_writer
        writer_with(Writer::Send::Regular, target)
      end
      memoize :target_writer

      def emit_target
        case target.type
        when :send
          emit_send_target
        when :lambda
          visit(target)
          emit_lambda_arguments unless node.type.equal?(:numblock)
        else
          visit(target)
        end
      end

      def emit_send_target
        target_writer.emit_receiver
        target_writer.emit_selector
        target_writer.emit_arguments_without_heredoc_body
      end

      def emit_lambda_arguments
        parentheses { writer_with(Args, arguments).emit_lambda_arguments }
      end

      def numblock?
        node.type.equal?(:numblock)
      end

      def emit_block_arguments
        return if numblock? || arguments.children.empty?

        ws

        parentheses('|', '|') do
          writer_with(Args, arguments).emit_block_arguments
        end
      end

    end # Block
  end # Emitter
end # Unparser
