# frozen_string_literal: true

module Unparser
  UnknownNodeError = Class.new(ArgumentError)

  # Emitter base class
  class Emitter
    include Adamantium, AbstractType, Constants, Generation, NodeHelpers
    include Anima.new(:buffer, :comments, :node, :local_variable_scope)

    public :node

    extend DSL

    # Registry for node emitters
    REGISTRY = {} # rubocop:disable Style/MutableConstant

    NO_INDENT = %i[ensure rescue].freeze

    module LocalVariableRoot
      # Return local variable root
      #
      # @return [Parser::AST::Node]
      #
      # @api private
      #
      def local_variable_scope
        AST::LocalVariableScope.new(node)
      end

      def self.included(descendant)
        descendant.class_eval do
          memoize :local_variable_scope
        end
      end
    end # LocalVariableRoot

    def node_type
      node.type
    end

    # Register emitter for type
    #
    # @param [Symbol] types
    #
    # @return [undefined]
    #
    # @api private
    #
    def self.handle(*types)
      types.each do |type|
        fail "Handler for type: #{type} already registered" if REGISTRY.key?(type)

        REGISTRY[type] = self
      end
    end
    private_class_method :handle

    def emit_mlhs
      dispatch
    end

    # Return emitter
    #
    # @return [Emitter]
    #
    # @api private
    #
    # rubocop:disable Metrics/ParameterLists
    def self.emitter(buffer:, comments:, node:, local_variable_scope:)
      type = node.type

      klass = REGISTRY.fetch(type) do
        fail UnknownNodeError, "Unknown node type: #{type.inspect}"
      end

      klass.new(
        buffer:               buffer,
        comments:             comments,
        local_variable_scope: local_variable_scope,
        node:                 node
      )
    end
    # rubocop:enable Metrics/ParameterLists

    # Dispatch node write as statement
    #
    # @return [undefined]
    #
    # @api private
    #
    abstract_method :dispatch

  end # Emitter
end # Unparser
