# frozen_string_literal: true

require 'diff/lcs'
require 'diff/lcs/hunk'
require 'optparse'
require 'parser/current'
require 'set'

require 'unparser/equalizer'
require 'unparser/adamantium'
require 'unparser/adamantium/method_builder'
require 'unparser/abstract_type'

require 'unparser/concord'
require 'unparser/either'
require 'unparser/anima'
require 'unparser/anima/attribute'
require 'unparser/anima/error'

# Library namespace
module Unparser
  # Unparser specific AST builder defaulting to modern AST format
  class Builder < Parser::Builders::Default
    modernize

    def initialize
      super

      self.emit_file_line_as_literals = false
    end
  end

  EMPTY_STRING = ''.freeze
  EMPTY_ARRAY  = [].freeze

  private_constant(*constants(false) - %i[Adamantium AbstractType Anima Concord Either Equalizer Memoizable])

  # Error raised when unparser encounters an invalid AST
  class InvalidNodeError < RuntimeError
    attr_reader :node

    def initialize(message, node)
      super(message)
      @node = node
      freeze
    end
  end

  # Unparse an AST (and, optionally, comments) into a string
  #
  # @param [Parser::AST::Node, nil] node
  # @param [Array] comment_array
  #
  # @return [String]
  #
  # @raise InvalidNodeError
  #   if the node passed is invalid
  #
  # @api public
  def self.unparse(node, comment_array = [])
    return '' if node.nil?

    Buffer.new.tap do |buffer|
      Emitter::Root.new(
        buffer,
        node,
        Comments.new(comment_array)
      ).write_to_buffer
    end.content
  end

  # Unparse with validation
  #
  # @param [Parser::AST::Node, nil] node
  # @param [Array] comment_array
  #
  # @return [Either<Validation,String>]
  def self.unparse_validate(node, comment_array = [])
    generated = unparse(node, comment_array)
    validation = Validation.from_string(generated)

    if validation.success?
      Either::Right.new(generated)
    else
      Either::Left.new(validation)
    end
  end

  # Unparse capturing errors
  #
  # This is mostly useful for writing testing tools against unparser.
  #
  # @param [Parser::AST::Node, nil] node
  #
  # @return [Either<Exception, String>]
  def self.unparse_either(node)
    Either.wrap_error(Exception) { unparse(node) }
  end

  # Parse string into AST
  #
  # @param [String] source
  #
  # @return [Parser::AST::Node, nil]
  def self.parse(source)
    parser.parse(buffer(source))
  end

  # Parse string into either syntax error or AST
  #
  # @param [String] source
  #
  # @return [Either<Parser::SyntaxError, (Parser::ASTNode, nil)>]
  def self.parse_either(source)
    Either.wrap_error(Parser::SyntaxError) do
      parser.parse(buffer(source))
    end
  end

  # Parse string into AST, with comments
  #
  # @param [String] source
  #
  # @return [Parser::AST::Node]
  def self.parse_with_comments(source)
    parser.parse_with_comments(buffer(source))
  end

  # Parser instance that produces AST unparser understands
  #
  # @return [Parser::Base]
  #
  # @api private
  def self.parser
    Parser::CurrentRuby.new(Builder.new).tap do |parser|
      parser.diagnostics.tap do |diagnostics|
        diagnostics.all_errors_are_fatal = true
      end
    end
  end

  # Construct a parser buffer from string
  #
  # @param [String] source
  #
  # @return [Parser::Source::Buffer]
  def self.buffer(source, identification = '(string)')
    Parser::Source::Buffer.new(identification, source: source)
  end
end # Unparser

require 'unparser/node_helpers'
require 'unparser/ast'
require 'unparser/ast/local_variable_scope'
require 'unparser/buffer'
require 'unparser/generation'
require 'unparser/color'
require 'unparser/comments'
require 'unparser/constants'
require 'unparser/diff'
require 'unparser/dsl'
require 'unparser/emitter'
require 'unparser/emitter/alias'
require 'unparser/emitter/args'
require 'unparser/emitter/argument'
require 'unparser/emitter/array'
require 'unparser/emitter/array_pattern'
require 'unparser/emitter/assignment'
require 'unparser/emitter/begin'
require 'unparser/emitter/binary'
require 'unparser/emitter/block'
require 'unparser/emitter/case'
require 'unparser/emitter/case_guard'
require 'unparser/emitter/case_match'
require 'unparser/emitter/cbase'
require 'unparser/emitter/class'
require 'unparser/emitter/const_pattern'
require 'unparser/emitter/def'
require 'unparser/emitter/defined'
require 'unparser/emitter/dstr'
require 'unparser/emitter/dsym'
require 'unparser/emitter/flipflop'
require 'unparser/emitter/float'
require 'unparser/emitter/flow_modifier'
require 'unparser/emitter/for'
require 'unparser/emitter/hash'
require 'unparser/emitter/hash_pattern'
require 'unparser/emitter/hookexe'
require 'unparser/emitter/if'
require 'unparser/emitter/in_match'
require 'unparser/emitter/in_pattern'
require 'unparser/emitter/index'
require 'unparser/emitter/kwbegin'
require 'unparser/emitter/lambda'
require 'unparser/emitter/masgn'
require 'unparser/emitter/match'
require 'unparser/emitter/match_alt'
require 'unparser/emitter/match_as'
require 'unparser/emitter/match_rest'
require 'unparser/emitter/match_var'
require 'unparser/emitter/mlhs'
require 'unparser/emitter/module'
require 'unparser/emitter/op_assign'
require 'unparser/emitter/pin'
require 'unparser/emitter/primitive'
require 'unparser/emitter/range'
require 'unparser/emitter/regexp'
require 'unparser/emitter/repetition'
require 'unparser/emitter/rescue'
require 'unparser/emitter/root'
require 'unparser/emitter/send'
require 'unparser/emitter/simple'
require 'unparser/emitter/splat'
require 'unparser/emitter/super'
require 'unparser/emitter/undef'
require 'unparser/emitter/variable'
require 'unparser/emitter/xstr'
require 'unparser/emitter/yield'
require 'unparser/emitter/kwargs'
require 'unparser/emitter/pair'
require 'unparser/emitter/match_pattern'
require 'unparser/writer'
require 'unparser/writer/binary'
require 'unparser/writer/dynamic_string'
require 'unparser/writer/resbody'
require 'unparser/writer/rescue'
require 'unparser/writer/send'
require 'unparser/writer/send/attribute_assignment'
require 'unparser/writer/send/binary'
require 'unparser/writer/send/regular'
require 'unparser/writer/send/unary'
require 'unparser/node_details'
require 'unparser/node_details/send'
require 'unparser/cli'

require 'unparser/validation'
# make it easy for zombie
require 'unparser/finalize'
