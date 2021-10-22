# typed: strict
# frozen_string_literal: true

module YARDSorbet
  # Helper methods for working with `YARD` AST Nodes
  module NodeUtils
    extend T::Sig

    # Command node types that can have type signatures
    ATTRIBUTE_METHODS = T.let(%i[attr attr_accessor attr_reader attr_writer].freeze, T::Array[Symbol])
    # Node types that can have type signatures
    SIGABLE_NODE = T.type_alias do
      T.any(YARD::Parser::Ruby::MethodDefinitionNode, YARD::Parser::Ruby::MethodCallNode)
    end
    # Skip these method contents during BFS node traversal, they can have their own nested types via `T.Proc`
    SKIP_METHOD_CONTENTS = T.let(%i[params returns], T::Array[Symbol])

    private_constant :ATTRIBUTE_METHODS, :SIGABLE_NODE

    # Traverese AST nodes in breadth-first order
    # @note This will skip over some node types.
    # @yield [YARD::Parser::Ruby::AstNode]
    sig do
      params(
        node: YARD::Parser::Ruby::AstNode,
        _blk: T.proc.params(n: YARD::Parser::Ruby::AstNode).void
      ).void
    end
    def self.bfs_traverse(node, &_blk)
      queue = [node]
      until queue.empty?
        n = T.must(queue.shift)
        yield n
        n.children.each { |c| queue.push(c) }
        queue.pop if n.is_a?(YARD::Parser::Ruby::MethodCallNode) && SKIP_METHOD_CONTENTS.include?(n.method_name(true))
      end
    end

    # Gets the node that a sorbet `sig` can be attached do, bypassing visisbility modifiers and the like
    sig { params(node: YARD::Parser::Ruby::AstNode).returns(SIGABLE_NODE) }
    def self.get_method_node(node)
      case node
      when YARD::Parser::Ruby::MethodDefinitionNode
        return node
      when YARD::Parser::Ruby::MethodCallNode
        return node if ATTRIBUTE_METHODS.include?(node.method_name(true))
      end

      node.jump(:def, :defs)
    end

    # Find and return the adjacent node (ascending)
    # @raise [IndexError] if the node does not have an adjacent sibling (ascending)
    sig { params(node: YARD::Parser::Ruby::AstNode).returns(YARD::Parser::Ruby::AstNode) }
    def self.sibling_node(node)
      siblings = node.parent.children
      siblings.each_with_index.find do |sibling, i|
        return siblings.fetch(i + 1) if sibling.equal?(node)
      end
    end
  end
end
