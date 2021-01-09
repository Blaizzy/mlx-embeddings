# typed: strict
# frozen_string_literal: true

require "ast_constants"
require "rubocop-ast"

module Utils
  # Helper functions for editing Ruby files.
  #
  # @api private
  module AST
    Node = RuboCop::AST::Node
    SendNode = RuboCop::AST::SendNode
    BlockNode = RuboCop::AST::BlockNode
    ProcessedSource = RuboCop::AST::ProcessedSource

    class << self
      extend T::Sig

      sig { params(body_node: Node).returns(T::Array[Node]) }
      def body_children(body_node)
        if body_node.nil?
          []
        elsif body_node.begin_type?
          body_node.children.compact
        else
          [body_node]
        end
      end

      sig { params(formula_contents: String).returns(T.nilable(Node)) }
      def bottle_block(formula_contents)
        formula_stanza(formula_contents, :bottle, type: :block_call)
      end

      sig { params(formula_contents: String, name: Symbol, type: T.nilable(Symbol)).returns(T.nilable(Node)) }
      def formula_stanza(formula_contents, name, type: nil)
        _, children = process_formula(formula_contents)
        children.find { |child| call_node_match?(child, name: name, type: type) }
      end

      sig { params(formula_contents: String, bottle_output: String).void }
      def replace_bottle_stanza!(formula_contents, bottle_output)
        replace_formula_stanza!(formula_contents, :bottle, bottle_output.chomp, type: :block_call)
      end

      sig { params(formula_contents: String, bottle_output: String).void }
      def add_bottle_stanza!(formula_contents, bottle_output)
        add_formula_stanza!(formula_contents, :bottle, "\n#{bottle_output.chomp}", type: :block_call)
      end

      sig do
        params(
          formula_contents: String,
          name:             Symbol,
          replacement:      T.any(Numeric, String, Symbol),
          type:             T.nilable(Symbol),
        ).void
      end
      def replace_formula_stanza!(formula_contents, name, replacement, type: nil)
        processed_source, children = process_formula(formula_contents)
        stanza_node = children.find { |child| call_node_match?(child, name: name, type: type) }
        raise "Could not find #{name} stanza!" if stanza_node.nil?

        tree_rewriter = Parser::Source::TreeRewriter.new(processed_source.buffer)
        tree_rewriter.replace(stanza_node.source_range, stanza_text(name, replacement, indent: 2).lstrip)
        formula_contents.replace(tree_rewriter.process)
      end

      sig do
        params(
          formula_contents: String,
          name:             Symbol,
          value:            T.any(Numeric, String, Symbol),
          type:             T.nilable(Symbol),
        ).void
      end
      def add_formula_stanza!(formula_contents, name, value, type: nil)
        processed_source, children = process_formula(formula_contents)

        preceding_component = if children.length > 1
          children.reduce do |previous_child, current_child|
            if formula_component_before_target?(current_child,
                                                target_name: name,
                                                target_type: type)
              next current_child
            else
              break previous_child
            end
          end
        else
          children.first
        end
        preceding_component = preceding_component.last_argument if preceding_component.is_a?(SendNode)

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
        tree_rewriter.insert_after(preceding_expr, "\n#{stanza_text(name, value, indent: 2)}")
        formula_contents.replace(tree_rewriter.process)
      end

      sig { params(name: Symbol, value: T.any(Numeric, String, Symbol), indent: T.nilable(Integer)).returns(String) }
      def stanza_text(name, value, indent: nil)
        text = if value.is_a?(String)
          _, node = process_source(value)
          value if (node.is_a?(SendNode) || node.is_a?(BlockNode)) && node.method_name == name
        end
        text ||= "#{name} #{value.inspect}"
        text = text.indent(indent) if indent && !text.match?(/\A\n* +/)
        text
      end

      private

      sig { params(source: String).returns([ProcessedSource, Node]) }
      def process_source(source)
        ruby_version = Version.new(HOMEBREW_REQUIRED_RUBY_VERSION).major_minor.to_f
        processed_source = ProcessedSource.new(source, ruby_version)
        root_node = processed_source.ast
        [processed_source, root_node]
      end

      sig { params(formula_contents: String).returns([ProcessedSource, T::Array[Node]]) }
      def process_formula(formula_contents)
        processed_source, root_node = process_source(formula_contents)

        class_node = root_node if root_node.class_type?
        if root_node.begin_type?
          nodes = root_node.children.select(&:class_type?)
          class_node = if nodes.count > 1
            nodes.find { |n| n.parent_class&.const_name == "Formula" }
          else
            nodes.first
          end
        end

        raise "Could not find formula class!" if class_node.nil?

        children = body_children(class_node.body)
        raise "Formula class is empty!" if children.empty?

        [processed_source, children]
      end

      sig { params(node: Node, target_name: Symbol, target_type: T.nilable(Symbol)).returns(T::Boolean) }
      def formula_component_before_target?(node, target_name:, target_type: nil)
        FORMULA_COMPONENT_PRECEDENCE_LIST.each do |components|
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

      sig do
        params(
          component_name: Symbol,
          component_type: Symbol,
          target_name:    Symbol,
          target_type:    T.nilable(Symbol),
        ).returns(T::Boolean)
      end
      def component_match?(component_name:, component_type:, target_name:, target_type: nil)
        component_name == target_name && (target_type.nil? || component_type == target_type)
      end

      sig { params(node: Node, name: Symbol, type: T.nilable(Symbol)).returns(T::Boolean) }
      def call_node_match?(node, name:, type: nil)
        node_type = case node
        when SendNode then :method_call
        when BlockNode then :block_call
        else return false
        end

        component_match?(component_name: node.method_name,
                         component_type: node_type,
                         target_name:    name,
                         target_type:    type)
      end
    end
  end
end
