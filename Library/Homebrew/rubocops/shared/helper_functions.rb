# typed: false
# frozen_string_literal: true

require "rubocop"

module RuboCop
  module Cop
    # Helper functions for cops.
    #
    # @api private
    module HelperFunctions
      include RangeHelp

      # Checks for regex match of pattern in the node and
      # sets the appropriate instance variables to report the match.
      def regex_match_group(node, pattern)
        string_repr = string_content(node).encode("UTF-8", invalid: :replace)
        match_object = string_repr.match(pattern)
        return unless match_object

        node_begin_pos = start_column(node)
        line_begin_pos = line_start_column(node)
        @column = if node_begin_pos == line_begin_pos
          node_begin_pos + match_object.begin(0) - line_begin_pos
        else
          node_begin_pos + match_object.begin(0) - line_begin_pos + 1
        end
        @length = match_object.to_s.length
        @line_no = line_number(node)
        @source_buf = source_buffer(node)
        @offensive_node = node
        match_object
      end

      # Returns the begin position of the node's line in source code.
      def line_start_column(node)
        node.source_range.source_buffer.line_range(node.loc.line).begin_pos
      end

      # Returns the begin position of the node in source code.
      def start_column(node)
        node.source_range.begin_pos
      end

      # Returns the line number of the node.
      def line_number(node)
        node.loc.line
      end

      # Source buffer is required as an argument to report style violations.
      def source_buffer(node)
        node.source_range.source_buffer
      end

      # Returns the string representation if node is of type str(plain) or dstr(interpolated) or const.
      def string_content(node)
        case node.type
        when :str
          node.str_content
        when :dstr
          content = ""
          node.each_child_node(:str, :begin) do |child|
            content += if child.begin_type?
              child.source
            else
              child.str_content
            end
          end
          content
        when :const
          node.const_name
        when :sym
          node.children.first.to_s
        else
          ""
        end
      end

      def problem(msg, &block)
        add_offense(@offensive_node, message: msg, &block)
      end
    end
  end
end
