# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for regexp literals
    class Regexp < self
      handle :regexp

      define_group(:body, 0..-2)

    private

      def dispatch
        parentheses('/', '/') do
          body.each(&method(:emit_body))
        end
        emit_options
      end

      def emit_options
        write(children.last.children.join)
      end

      def emit_body(node)
        if n_begin?(node)
          write('#{')
          node.children.each(&method(:visit))
          write('}')
        else
          buffer.append_without_prefix(node.children.first.gsub('/', '\/'))
        end
      end
    end # Regexp
  end # Emitter
end # Unparser
