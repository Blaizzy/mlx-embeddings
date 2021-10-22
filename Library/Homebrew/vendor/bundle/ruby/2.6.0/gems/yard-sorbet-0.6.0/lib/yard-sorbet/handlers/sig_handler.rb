# typed: strict
# frozen_string_literal: true

module YARDSorbet
  module Handlers
    # A YARD Handler for Sorbet type declarations
    class SigHandler < YARD::Handlers::Ruby::Base
      extend T::Sig

      handles method_call(:sig)
      namespace_only

      # These node types attached to sigs represent attr_* declarations
      ATTR_NODE_TYPES = T.let(%i[command fcall], T::Array[Symbol])
      private_constant :ATTR_NODE_TYPES

      # Swap the method definition docstring and the sig docstring.
      # Parse relevant parts of the `sig` and include them as well.
      sig { void }
      def process
        method_node = NodeUtils.get_method_node(NodeUtils.sibling_node(statement))
        docstring, directives = Directives.extract_directives(statement.docstring)
        parse_sig(method_node, docstring)
        method_node.docstring = docstring.to_raw
        Directives.add_directives(method_node.docstring, directives)
        statement.docstring = nil
      end

      private

      sig { params(method_node: YARD::Parser::Ruby::AstNode, docstring: YARD::Docstring).void }
      def parse_sig(method_node, docstring)
        NodeUtils.bfs_traverse(statement) do |n|
          case n.source
          when 'abstract'
            YARDSorbet::TagUtils.upsert_tag(docstring, 'abstract')
          when 'params'
            parse_params(method_node, n, docstring)
          when 'returns', 'void'
            parse_return(n, docstring)
          end
        end
      end

      sig do
        params(
          method_node: YARD::Parser::Ruby::AstNode,
          node: YARD::Parser::Ruby::AstNode,
          docstring: YARD::Docstring
        ).void
      end
      def parse_params(method_node, node, docstring)
        return if ATTR_NODE_TYPES.include?(method_node.type)

        sibling = NodeUtils.sibling_node(node)
        sibling[0][0].each do |p|
          param_name = p[0][0]
          types = SigToYARD.convert(p.last)
          TagUtils.upsert_tag(docstring, 'param', types, param_name)
        end
      end

      sig { params(node: YARD::Parser::Ruby::AstNode, docstring: YARD::Docstring).void }
      def parse_return(node, docstring)
        type = node.source == 'void' ? ['void'] : SigToYARD.convert(NodeUtils.sibling_node(node))
        TagUtils.upsert_tag(docstring, 'return', type)
      end
    end
  end
end
