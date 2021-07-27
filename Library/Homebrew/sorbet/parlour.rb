# frozen_string_literal: true
# typed: true

require_relative "../warnings"
Warnings.ignore :parser_syntax do
  require "parser/current"
end

module Homebrew
  module Parlour
    extend T::Sig

    ROOT_DIR = T.let(Pathname(__dir__).parent.realpath.freeze, Pathname)

    sig { returns(T::Array[Parser::AST::Node]) }
    def self.ast_list
      @@ast_list ||= begin
        ast_list = []
        parser = Parser::CurrentRuby.new

        ROOT_DIR.find do |path|
          Find.prune if path.directory? && %w[sorbet shims test vendor].any? { |subdir| path == ROOT_DIR/subdir }

          Find.prune if path.file? && path.extname != ".rb"

          next unless path.file?

          buffer = Parser::Source::Buffer.new(path, source: path.read)

          parser.reset
          ast = parser.parse(buffer)
          ast_list << ast if ast
        end

        ast_list
      end
    end
  end
end

require "parlour"
require_relative "parlour/attr"
