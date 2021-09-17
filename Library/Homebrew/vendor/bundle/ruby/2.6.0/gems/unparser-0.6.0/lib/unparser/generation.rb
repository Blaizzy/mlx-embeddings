# frozen_string_literal: true

module Unparser
  # rubocop:disable Metrics/ModuleLength
  module Generation
    EXTRA_NL = %i[kwbegin def defs module class sclass].freeze

    private_constant(*constants(false))

    def emit_heredoc_reminders; end

    def symbol_name; end

    def write_to_buffer
      with_comments { dispatch }
      self
    end

  private

    def delimited(nodes, delimiter = ', ', &block)
      return if nodes.empty?

      emit_join(nodes, block || method(:visit), -> { write(delimiter) })
    end

    def emit_join(nodes, emit_node, emit_delimiter)
      return if nodes.empty?

      head, *tail = nodes
      emit_node.call(head)

      tail.each do |node|
        emit_delimiter.call
        emit_node.call(node)
      end
    end

    def nl
      emit_eol_comments
      buffer.nl
    end

    def with_comments
      emit_comments_before if buffer.fresh_line?
      yield
      comments.consume(node)
    end

    def ws
      write(' ')
    end

    def emit_eol_comments
      comments.take_eol_comments.each do |comment|
        write(' ', comment.text)
      end
    end

    def emit_eof_comments
      emit_eol_comments
      comments_left = comments.take_all
      return if comments_left.empty?

      buffer.nl
      emit_comments(comments_left)
    end

    def emit_comments_before(source_part = :expression)
      comments_before = comments.take_before(node, source_part)
      return if comments_before.empty?

      emit_comments(comments_before)
      buffer.nl
    end

    def emit_comments(comments)
      max = comments.size - 1
      comments.each_with_index do |comment, index|
        if comment.type.equal?(:document)
          buffer.append_without_prefix(comment.text.chomp)
        else
          write(comment.text)
        end
        buffer.nl if index < max
      end
    end

    def write(*strings)
      strings.each(&buffer.method(:append))
    end

    def k_end
      buffer.indent
      emit_comments_before(:end)
      buffer.unindent
      write('end')
    end

    def parentheses(open = '(', close = ')')
      write(open)
      yield
      write(close)
    end

    def indented
      buffer = buffer()
      buffer.indent
      nl
      yield
      nl
      buffer.unindent
    end

    def emit_optional_body(node, indent: true)
      if node
        emit_body(node, indent: indent)
      else
        nl
      end
    end

    def emit_body(node, indent: true)
      if indent
        buffer.indent
        nl
      end

      if n_begin?(node)
        if node.children.one?
          visit_deep(node)
        else
          emit_body_inner(node)
        end
      else
        visit_deep(node)
      end

      if indent
        buffer.unindent
        nl
      end
    end

    def emit_body_inner(node)
      head, *tail = node.children
      emit_body_member(head)

      tail.each do |child|
        nl

        nl if EXTRA_NL.include?(child.type)

        emit_body_member(child)
      end
    end

    def emit_body_member(node)
      if n_rescue?(node)
        emit_rescue_postcontrol(node)
      else
        visit_deep(node)
      end
    end

    def emit_ensure(node)
      body, ensure_body = node.children

      if body
        emit_body_rescue(body)
      else
        nl
      end

      write('ensure')

      emit_optional_body(ensure_body)
    end

    def emit_body_rescue(node)
      if n_rescue?(node)
        emit_rescue_regular(node)
      else
        emit_body(node)
      end
    end

    def emit_optional_body_ensure_rescue(node)
      if node
        emit_body_ensure_rescue(node)
      else
        nl
      end
    end

    def emit_body_ensure_rescue(node)
      if n_ensure?(node)
        emit_ensure(node)
      elsif n_rescue?(node)
        emit_rescue_regular(node)
      else
        emit_body(node)
      end
    end

    def emit_rescue_postcontrol(node)
      writer = writer_with(Writer::Rescue, node)
      writer.emit_postcontrol
      writer.emit_heredoc_reminders
    end

    def emit_rescue_regular(node)
      writer_with(Writer::Rescue, node).emit_regular
    end

    def writer_with(klass, node)
      klass.new(to_h.merge(node: node))
    end

    def emitter(node)
      Emitter.emitter(**to_h.merge(node: node))
    end

    def visit(node)
      emitter(node).write_to_buffer
    end

    def visit_deep(node)
      emitter(node).tap do |emitter|
        emitter.write_to_buffer
        emitter.emit_heredoc_reminders
      end
    end

    def first_child
      children.first
    end

    def conditional_parentheses(flag, &block)
      if flag
        parentheses(&block)
      else
        block.call
      end
    end

    def children
      node.children
    end
  end # Generation
  # rubocop:enable Metrics/ModuleLength
end # Unparser
