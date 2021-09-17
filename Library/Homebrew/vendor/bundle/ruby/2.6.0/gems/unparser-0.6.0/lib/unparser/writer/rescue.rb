# frozen_string_literal: true

module Unparser
  module Writer
    class Rescue
      include Writer, Adamantium

      children :body, :rescue_body

      define_group :rescue_bodies, 1..-2

      def emit_regular
        emit_optional_body(body)

        rescue_bodies.each(&method(:emit_rescue_body))

        if else_node
          write('else')
          emit_body(else_node)
        end
      end

      def emit_heredoc_reminders
        emitter(body).emit_heredoc_reminders
      end

      def emit_postcontrol
        visit(body)
        writer_with(Resbody, rescue_body).emit_postcontrol
      end

    private

      def else_node
        children.last
      end

      def emit_rescue_body(node)
        writer_with(Resbody, node).emit_regular
      end
    end # Rescue
  end # Writer
end # Unparser
