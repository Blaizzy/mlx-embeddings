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
    TreeRewriter = Parser::Source::TreeRewriter

    # Helper class for editing formulae.
    #
    # @api private
    class FormulaAST
      extend T::Sig
      extend Forwardable

      delegate process: :tree_rewriter

      sig { params(formula_contents: String).void }
      def initialize(formula_contents)
        @formula_contents = formula_contents
        processed_source, children = process_formula
        @processed_source = T.let(processed_source, ProcessedSource)
        @children = T.let(children, T::Array[Node])
        @tree_rewriter = T.let(TreeRewriter.new(processed_source.buffer), TreeRewriter)
      end

      sig { params(body_node: Node).returns(T::Array[Node]) }
      def self.body_children(body_node)
        if body_node.nil?
          []
        elsif body_node.begin_type?
          body_node.children.compact
        else
          [body_node]
        end
      end

      sig { returns(T.nilable(Node)) }
      def bottle_block
        stanza(:bottle, type: :block_call)
      end

      sig { params(name: Symbol, type: T.nilable(Symbol)).returns(T.nilable(Node)) }
      def stanza(name, type: nil)
        children.find { |child| call_node_match?(child, name: name, type: type) }
      end

      sig { params(bottle_output: String).void }
      def replace_bottle_block(bottle_output)
        replace_stanza(:bottle, bottle_output.chomp, type: :block_call)
      end

      sig { params(bottle_output: String).void }
      def add_bottle_block(bottle_output)
        add_stanza(:bottle, "\n#{bottle_output.chomp}", type: :block_call)
      end

      sig { params(name: Symbol, replacement: T.any(Numeric, String, Symbol), type: T.nilable(Symbol)).void }
      def replace_stanza(name, replacement, type: nil)
        stanza_node = children.find { |child| call_node_match?(child, name: name, type: type) }
        raise "Could not find #{name} stanza!" if stanza_node.nil?

        tree_rewriter.replace(stanza_node.source_range, self.class.stanza_text(name, replacement, indent: 2).lstrip)
      end

      sig { params(name: Symbol, value: T.any(Numeric, String, Symbol), type: T.nilable(Symbol)).void }
      def add_stanza(name, value, type: nil)
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

        tree_rewriter.insert_after(preceding_expr, "\n#{self.class.stanza_text(name, value, indent: 2)}")
      end

      sig { params(name: Symbol, value: T.any(Numeric, String, Symbol), indent: T.nilable(Integer)).returns(String) }
      def self.stanza_text(name, value, indent: nil)
        text = if value.is_a?(String)
          _, node = process_source(value)
          value if (node.is_a?(SendNode) || node.is_a?(BlockNode)) && node.method_name == name
        end
        text ||= "#{name} #{value.inspect}"
        text = text.indent(indent) if indent && !text.match?(/\A\n* +/)
        text
      end

      sig { params(source: String).returns([ProcessedSource, Node]) }
      def self.process_source(source)
        ruby_version = Version.new(HOMEBREW_REQUIRED_RUBY_VERSION).major_minor.to_f
        processed_source = ProcessedSource.new(source, ruby_version)
        root_node = processed_source.ast
        [processed_source, root_node]
      end

      private

      sig { returns(String) }
      attr_reader :formula_contents

      sig { returns(ProcessedSource) }
      attr_reader :processed_source

      sig { returns(T::Array[Node]) }
      attr_reader :children

      sig { returns(TreeRewriter) }
      attr_reader :tree_rewriter

      sig { returns([ProcessedSource, T::Array[Node]]) }
      def process_formula
        processed_source, root_node = self.class.process_source(formula_contents)

        class_node = if root_node.class_type?
          root_node
        elsif root_node.begin_type?
          root_node.children.find { |n| n.class_type? && n.parent_class&.const_name == "Formula" }
        end

        raise "Could not find formula class!" if class_node.nil?

        children = self.class.body_children(class_node.body)
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
