# frozen_string_literal: true

module Unparser
  class Emitter
    # Dynamic symbol literal emitter
    class DSym < self
      handle :dsym

    private

      def dispatch
        write(':"')
        children.each do |child|
          case child.type
          when :str
            emit_str_child(child)
          when :begin
            emit_begin_child(child)
          end
        end
        write('"')
      end

      def emit_str_child(value)
        string = value.children.first
        if string.end_with?("\n")
          write(string.inspect[1..-4])
          nl
        else
          write(string.inspect[1..-2])
        end
      end

      def emit_begin_child(component)
        write('#{')
        visit(unwrap_single_begin(component))
        write('}')
      end
    end # DSym
  end # Emitter
end # Unparser
