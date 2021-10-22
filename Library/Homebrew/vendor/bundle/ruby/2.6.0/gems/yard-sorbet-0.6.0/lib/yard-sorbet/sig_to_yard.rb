# typed: strict
# frozen_string_literal: true

module YARDSorbet
  # Translate `sig` type syntax to `YARD` type syntax.
  module SigToYARD
    extend T::Sig

    # @see https://yardoc.org/types.html
    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    def self.convert(node)
      # scrub newlines, as they break the YARD parser
      convert_node(node).map { |type| type.gsub(/\n\s*/, ' ') }
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_node(node)
      case node
      when YARD::Parser::Ruby::MethodCallNode then convert_call(node)
      when YARD::Parser::Ruby::ReferenceNode then convert_ref(node)
      else convert_node_type(node)
      end
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_node_type(node)
      case node.type
      when :aref then convert_aref(node)
      when :arg_paren then convert_node(node.first)
      when :array then convert_array(node)
      # Fixed hashes as return values are unsupported:
      # https://github.com/lsegal/yard/issues/425
      #
      # Hash key params can be individually documented with `@option`, but
      # sig translation is currently unsupported.
      when :hash then ['Hash']
      # seen when sig methods omit parentheses
      when :list then convert_list(node)
      else convert_unknown(node)
      end
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(String) }
    private_class_method def self.build_generic_type(node)
      return node.source if node.empty? || node.type != :aref

      collection_type = node.first.source
      member_type = node.last.children.map { |child| build_generic_type(child) }.join(', ')

      "#{collection_type}[#{member_type}]"
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_aref(node)
      # https://www.rubydoc.info/gems/yard/file/docs/Tags.md#Parametrized_Types
      case node.first.source
      when 'T::Array', 'T::Enumerable', 'T::Range', 'T::Set' then convert_collection(node)
      when 'T::Hash' then convert_hash(node)
      else
        log.info("Unsupported sig aref node #{node.source}")
        [build_generic_type(node)]
      end
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_array(node)
      # https://www.rubydoc.info/gems/yard/file/docs/Tags.md#Order-Dependent_Lists
      member_types = node.first.children.map { |n| convert_node(n) }
      sequence = member_types.map { |mt| mt.size == 1 ? mt[0] : mt.to_s.tr('"', '') }.join(', ')
      ["Array(#{sequence})"]
    end

    sig { params(node: YARD::Parser::Ruby::MethodCallNode).returns(T::Array[String]) }
    private_class_method def self.convert_call(node)
      node.namespace.source == 'T' ? convert_t_method(node) : [node.source]
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_collection(node)
      collection_type = node.first.source.split('::').last
      member_type = convert_node(node.last.first).join(', ')
      ["#{collection_type}<#{member_type}>"]
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_hash(node)
      kv = node.last.children
      key_type = convert_node(kv.first).join(', ')
      value_type = convert_node(kv.last).join(', ')
      ["Hash{#{key_type} => #{value_type}}"]
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_list(node)
      node.children.size == 1 ? convert_node(node.children.first) : [node.source]
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_ref(node)
      source = node.source
      case source
      when 'T::Boolean' then ['Boolean'] # YARD convention for booleans
      # YARD convention is use singleton objects when applicable:
      # https://www.rubydoc.info/gems/yard/file/docs/Tags.md#Literals
      when 'FalseClass' then ['false']
      when 'NilClass' then ['nil']
      when 'TrueClass' then ['true']
      else [source]
      end
    end

    sig { params(node: YARD::Parser::Ruby::MethodCallNode).returns(T::Array[String]) }
    private_class_method def self.convert_t_method(node)
      case node.method_name(true)
      when :any then node.last.first.children.map { |n| convert_node(n) }.flatten
      # Order matters here, putting `nil` last results in a more concise
      # return syntax in the UI (superscripted `?`)
      # https://github.com/lsegal/yard/blob/cfa62ae/lib/yard/templates/helpers/html_helper.rb#L499-L500
      when :nilable then convert_node(node.last) + ['nil']
      else [node.source]
      end
    end

    sig { params(node: YARD::Parser::Ruby::AstNode).returns(T::Array[String]) }
    private_class_method def self.convert_unknown(node)
      log.warn("Unsupported sig #{node.type} node #{node.source}")
      [node.source]
    end
  end
end
