# typed: strict
# frozen_string_literal: true

module YARDSorbet
  module Handlers
    # Handles all `const`/`prop` calls, creating accessor methods, and compiles them for later usage at the class level
    # in creating a constructor
    class StructPropHandler < YARD::Handlers::Ruby::Base
      extend T::Sig

      handles method_call(:const), method_call(:prop)
      namespace_only

      sig { void }
      def process
        name = statement.parameters.first.last.last.source
        prop = make_prop(name)
        update_state(prop)
        object = YARD::CodeObjects::MethodObject.new(namespace, name, scope)
        decorate_object(object, prop)
        register_attrs(object, name)
      end

      private

      # Add the source and docstring to the method object
      sig { params(object: YARD::CodeObjects::MethodObject, prop: TStructProp).void }
      def decorate_object(object, prop)
        object.source = prop.source
        # TODO: this should use `+` to delimit the attribute name when markdown is disabled
        reader_docstring = prop.doc.empty? ? "Returns the value of attribute `#{prop.prop_name}`." : prop.doc
        docstring = YARD::DocstringParser.new.parse(reader_docstring).to_docstring
        docstring.add_tag(YARD::Tags::Tag.new(:return, '', prop.types))
        object.docstring = docstring.to_raw
      end

      # Get the default prop value
      sig { returns(T.nilable(String)) }
      def default_value
        default_node = statement.traverse { |n| break n if n.type == :label && n.source == 'default:' }
        default_node.parent[1].source if default_node
      end

      sig { params(name: String).returns(TStructProp) }
      def make_prop(name)
        TStructProp.new(
          default: default_value,
          doc: statement.docstring.to_s,
          prop_name: name,
          source: statement.source,
          types: SigToYARD.convert(statement.parameters[1])
        )
      end

      # Register the field explicitly as an attribute.
      # While `const` attributes are immutable, `prop` attributes may be reassigned.
      sig { params(object: YARD::CodeObjects::MethodObject, name: String).void }
      def register_attrs(object, name)
        # Create the virtual method in our current scope
        write = statement.method_name(true) == :prop ? object : nil
        namespace.attributes[scope][name] ||= SymbolHash[read: object, write: write]
      end

      # Store the prop for use in the constructor definition
      sig { params(prop: TStructProp).void }
      def update_state(prop)
        extra_state.prop_docs ||= Hash.new { |h, k| h[k] = [] }
        extra_state.prop_docs[namespace] << prop
      end
    end
  end
end
