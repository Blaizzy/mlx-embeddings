# frozen_string_literal: true

module Unparser
  module NodeHelpers

    # Helper for building nodes
    #
    # @param [Symbol] type
    # @param [Parser::AST::Node] children
    #
    # @return [Parser::AST::Node]
    #
    # @api private
    def s(type, *children)
      Parser::AST::Node.new(type, children)
    end

    # Helper for building nodes
    #
    # @param [Symbol] type
    #
    # @return [Parser::AST::Node]
    # @param [Array] children
    #
    # @api private
    def n(type, children = [])
      Parser::AST::Node.new(type, children)
    end

    def n?(type, node)
      node.type.equal?(type)
    end

    %i[
      arg
      args
      array
      array_pattern
      empty_else
      begin
      block
      cbase
      const
      dstr
      ensure
      hash
      hash_pattern
      if
      in_pattern
      int
      kwsplat
      lambda
      match_rest
      pair
      rescue
      send
      shadowarg
      splat
      str
      sym
    ].each do |type|
      name = "n_#{type}?"
      define_method(name) do |node|
        n?(type, node)
      end
      private(name)
    end

    def unwrap_single_begin(node)
      if n_begin?(node) && node.children.one?
        node.children.first
      else
        node
      end
    end
  end # NodeHelpers
end # Unparser
