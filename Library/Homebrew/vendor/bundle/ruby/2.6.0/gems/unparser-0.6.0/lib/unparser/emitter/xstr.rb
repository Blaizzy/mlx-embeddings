# frozen_string_literal: true

module Unparser
  class Emitter
    # Dynamic execute string literal emitter
    class XStr < self

      handle :xstr

    private

      def dispatch
        if heredoc?
          emit_heredoc
        else
          emit_xstr
        end
      end

      def heredoc?
        children.any? { |node| node.eql?(s(:str, '')) }
      end

      def emit_heredoc
        write(%(<<~`HEREDOC`))
        buffer.indent
        nl
        children.each do |child|
          if n_str?(child)
            write(child.children.first)
          else
            emit_begin(child)
          end
        end
        buffer.unindent
        write("HEREDOC\n")
      end

      def emit_xstr
        write('`')
        children.each do |child|
          if n_begin?(child)
            emit_begin(child)
          else
            emit_string(child)
          end
        end
        write('`')
      end

      def emit_string(value)
        write(escape_xstr(value.children.first))
      end

      def escape_xstr(input)
        input.chars.map do |char|
          if char.eql?('`')
            '\\`'
          else
            char
          end
        end.join
      end

      def emit_begin(component)
        write('#{')
        visit(unwrap_single_begin(component))
        write('}')
      end
    end # XStr
  end # Emitter
end # Unparser
