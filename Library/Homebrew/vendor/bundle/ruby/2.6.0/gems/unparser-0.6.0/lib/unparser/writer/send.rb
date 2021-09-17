# frozen_string_literal: true

module Unparser
  module Writer
    # Writer for send
    class Send
      include Writer, Adamantium, Constants, Generation

      INDEX_ASSIGN    = :'[]='
      INDEX_REFERENCE = :'[]'

      OPERATORS = {
        csend: '&.',
        send:  '.'
      }.freeze

      private_constant(*constants(false))

      children :receiver, :selector

      def dispatch
        effective_writer.dispatch
      end

      def emit_mlhs
        effective_writer.emit_send_mlhs
      end

      def emit_selector
        write(details.string_selector)
      end

      def emit_heredoc_reminders
        emitter(receiver).emit_heredoc_reminders if receiver
        arguments.each(&method(:emit_heredoc_reminder))
      end

    private

      def effective_writer
        writer_with(effective_writer_class, node)
      end
      memoize :effective_writer

      def effective_writer_class
        if details.binary_syntax_allowed?
          Binary
        elsif details.selector_unary_operator? && n_send?(node)
          Unary
        elsif write_as_attribute_assignment?
          AttributeAssignment
        else
          Regular
        end
      end

      def write_as_attribute_assignment?
        details.assignment_operator?
      end

      def emit_operator
        write(OPERATORS.fetch(node.type))
      end

      def emit_arguments
        if arguments.empty?
          write('()') if receiver.nil? && avoid_clash?
        else
          emit_normal_arguments
        end
      end

      def arguments
        details.arguments
      end

      def emit_normal_arguments
        parentheses { delimited(arguments) }
      end

      def emit_heredoc_reminder(argument)
        emitter(argument).emit_heredoc_reminders
      end

      def avoid_clash?
        local_variable_clash? || parses_as_constant?
      end

      def local_variable_clash?
        local_variable_scope.local_variable_defined_for_node?(node, selector)
      end

      def parses_as_constant?
        test = Unparser.parse_either(selector.to_s).from_right do
          fail InvalidNodeError.new("Invalid selector for send node: #{selector.inspect}", node)
        end

        n_const?(test)
      end

      def details
        NodeDetails::Send.new(node)
      end
      memoize :details

      def emit_send_regular(node)
        if n_send?(node)
          writer_with(Regular, node).dispatch
        else
          visit(node)
        end
      end
    end # Send
  end # Writer
end # Unparser
