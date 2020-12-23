# typed: true
# frozen_string_literal: true

module Utils
  # Helper functions for editing Ruby files.
  #
  # @api private
  module AST
    class << self
      extend T::Sig

      def replace_formula_stanza!(formula_contents, name:, replacement:, type: nil)
        processed_source, body_node = process_formula(formula_contents)
        children = body_node.begin_type? ? body_node.children.compact : [body_node]
        stanza_node = children.find { |child| call_node_match?(child, name: name, type: type) }
        raise "Could not find #{name} stanza!" if stanza_node.nil?

        tree_rewriter = Parser::Source::TreeRewriter.new(processed_source.buffer)
        tree_rewriter.replace(stanza_node.source_range, replacement.strip)
        formula_contents.replace(tree_rewriter.process)
      end

      def add_formula_stanza!(formula_contents, name:, text:, type: nil)
        processed_source, body_node = process_formula(formula_contents)

        preceding_component = if body_node.begin_type?
          body_node.children.compact.reduce do |previous_child, current_child|
            if formula_component_before_target?(current_child,
                                                target_name: name,
                                                target_type: type)
              next current_child
            else
              break previous_child
            end
          end
        else
          body_node
        end
        preceding_component = preceding_component.last_argument if preceding_component.send_type?

        preceding_expr = preceding_component.location.expression
        processed_source.comments.each do |comment|
          comment_expr = comment.location.expression
          distance = comment_expr.first_line - preceding_expr.first_line
          case distance
          when 0
            if comment_expr.last_line > preceding_expr.last_line ||
               comment_expr.end_pos > preceding_expr.end_pos
              preceding_expr = comment_expr
            end
          when 1
            preceding_expr = comment_expr
          end
        end

        tree_rewriter = Parser::Source::TreeRewriter.new(processed_source.buffer)
        tree_rewriter.insert_after(preceding_expr, "\n#{text.match?(/\A\s+/) ? text : text.indent(2)}")
        formula_contents.replace(tree_rewriter.process)
      end

      private

      def process_formula(formula_contents)
        Homebrew.install_bundler_gems!
        require "rubocop-ast"

        ruby_version = Version.new(HOMEBREW_REQUIRED_RUBY_VERSION).major_minor.to_f
        processed_source = RuboCop::AST::ProcessedSource.new(formula_contents, ruby_version)
        root_node = processed_source.ast

        class_node = if root_node.class_type?
          root_node
        elsif root_node.begin_type?
          root_node.children.find { |n| n.class_type? && n.parent_class&.const_name == "Formula" }
        end

        raise "Could not find formula class!" if class_node.nil?

        body_node = class_node.body
        raise "Formula class is empty!" if body_node.nil?

        [processed_source, body_node]
      end

      def formula_component_before_target?(node, target_name:, target_type: nil)
        require "rubocops/components_order"

        RuboCop::Cop::FormulaAudit::ComponentsOrder::COMPONENT_PRECEDENCE_LIST.each do |components|
          return false if components.any? do |component|
            component_match?(component_name: component[:name],
                             component_type: component[:type],
                             target_name:    target_name,
                             target_type:    target_type)
          end
          return true if components.any? do |component|
            call_node_match?(node, name: component[:name], type: component[:type])
          end
        end

        false
      end

      def component_match?(component_name:, component_type:, target_name:, target_type: nil)
        component_name == target_name && (target_type.nil? || component_type == target_type)
      end

      def call_node_match?(node, name:, type: nil)
        node_type = if node.send_type?
          :method_call
        elsif node.block_type?
          :block_call
        end
        return false if node_type.nil?

        component_match?(component_name: T.unsafe(node).method_name,
                         component_type: node_type,
                         target_name:    name,
                         target_type:    type)
      end
    end
  end
end
