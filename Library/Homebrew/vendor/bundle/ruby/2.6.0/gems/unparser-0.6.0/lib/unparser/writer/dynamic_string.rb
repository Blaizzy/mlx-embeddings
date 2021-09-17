# frozen_string_literal: true

module Unparser
  module Writer
    class DynamicString
      include Writer, Adamantium

      PATTERNS_2 =
        [
          %i[str_empty begin].freeze,
          %i[begin str_nl].freeze
        ].freeze

      PATTERNS_3 =
        [
          %i[begin str_nl_eol str_nl_eol].freeze,
          %i[str_nl_eol begin str_nl_eol].freeze,
          %i[str_ws begin str_nl_eol].freeze
        ].freeze

      FLAT_INTERPOLATION = %i[ivar cvar gvar nth_ref].to_set.freeze

      private_constant(*constants(false))

      def emit_heredoc_reminder
        return unless heredoc?

        emit_heredoc_body
        emit_heredoc_footer
      end

      def dispatch
        if heredoc?
          emit_heredoc_header
        else
          emit_dstr
        end
      end

    private

      def heredoc_header
        '<<-HEREDOC'
      end

      def heredoc?
        !children.empty? && (nl_last_child? && heredoc_pattern?)
      end

      def emit_heredoc_header
        write(heredoc_header)
      end

      def emit_heredoc_body
        nl
        emit_normal_heredoc_body
      end

      def emit_heredoc_footer
        write('HEREDOC')
      end

      def classify(node)
        if n_str?(node)
          classify_str(node)
        else
          node.type
        end
      end

      def classify_str(node)
        if str_nl?(node)
          :str_nl
        elsif node.children.first.end_with?("\n")
          :str_nl_eol
        elsif str_ws?(node)
          :str_ws
        elsif str_empty?(node)
          :str_empty
        end
      end

      def str_nl?(node)
        node.eql?(s(:str, "\n"))
      end

      def str_empty?(node)
        node.eql?(s(:str, ''))
      end

      def str_ws?(node)
        /\A( |\t)+\z/.match?(node.children.first)
      end

      def heredoc_pattern?
        heredoc_pattern_2? || heredoc_pattern_3?
      end

      def heredoc_pattern_3?
        children.each_cons(3).any? do |group|
          PATTERNS_3.include?(group.map(&method(:classify)))
        end
      end

      def heredoc_pattern_2?
        children.each_cons(2).any? do |group|
          PATTERNS_2.include?(group.map(&method(:classify)))
        end
      end

      def nl_last_child?
        last = children.last
        n_str?(last) && last.children.first[-1].eql?("\n")
      end

      def emit_squiggly_heredoc_body
        buffer.indent
        children.each do |child|
          if n_str?(child)
            write(escape_dynamic(child.children.first))
          else
            emit_dynamic(child)
          end
        end
        buffer.unindent
      end

      def emit_normal_heredoc_body
        buffer.root_indent do
          children.each do |child|
            if n_str?(child)
              write(escape_dynamic(child.children.first))
            else
              emit_dynamic(child)
            end
          end
        end
      end

      def escape_dynamic(string)
        string.gsub('#', '\#')
      end

      def emit_dynamic(child)
        if FLAT_INTERPOLATION.include?(child.type)
          write('#')
          visit(child)
        elsif n_dstr?(child)
          emit_body(child.children)
        else
          write('#{')
          emit_dynamic_component(child.children.first)
          write('}')
        end
      end

      def emit_dynamic_component(node)
        visit(node) if node
      end

      def emit_dstr
        if children.empty?
          write('%()')
        else
          segments.each_with_index do |children, index|
            emit_segment(children, index)
          end
        end
      end

      def breakpoint?(child, current)
        last_type = current.last&.type

        [
          n_str?(child) && last_type.equal?(:str) && current.none?(&method(:n_begin?)),
          last_type.equal?(:dstr),
          n_dstr?(child) && last_type
        ].any?
      end

      def segments
        segments = []

        segments << current = []

        children.each do |child|
          if breakpoint?(child, current)
            segments << current = []
          end

          current << child
        end

        segments
      end

      def emit_segment(children, index)
        write(' ') unless index.zero?

        write('"')
        emit_body(children)
        write('"')
      end

      def emit_body(children)
        buffer.root_indent do
          children.each_with_index do |child, index|
            if n_str?(child)
              string = child.children.first
              if string.eql?("\n") && children.fetch(index.pred).type.equal?(:begin)
                write("\n")
              else
                write(string.inspect[1..-2])
              end
            else
              emit_dynamic(child)
            end
          end
        end
      end
    end # DynamicString
  end # Writer
end # Unparser
