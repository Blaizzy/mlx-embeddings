# frozen_string_literal: true

module Unparser
  module Writer
    # Writer for rescue bodies
    class Resbody
      include Writer

      children :exception, :assignment, :body

      def emit_postcontrol
        write(' rescue ')
        visit(body)
      end

      def emit_regular
        write('rescue')
        emit_exception
        emit_assignment
        emit_optional_body(body)
      end

    private

      def emit_exception
        return unless exception

        ws
        delimited(exception.children)
      end

      def emit_assignment
        return unless assignment

        write(' => ')
        visit(assignment)
      end
    end # Resbody
  end # Writer
end # Unparser
