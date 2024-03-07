# typed: strong
# frozen_string_literal: true

module Homebrew
  # Subclass this to implement a `brew` command. This is preferred to declaring a named function in the `Homebrew`
  # module, because:
  # - Each Command lives in an isolated namespace.
  # - Each Command implements a defined interface.
  #
  # To subclass, implement a `run` method and provide a `cmd_args` block to document the command and its allowed args.
  class AbstractCommand
    extend T::Helpers

    abstract!

    class << self
      sig { returns(T.nilable(T.proc.void)) }
      attr_reader :parser_block

      sig { returns(String) }
      def command_name = T.must(name).split("::").fetch(-1).downcase

      # @return the AbstractCommand subclass associated with the brew CLI command name.
      sig { params(name: String).returns(T.nilable(T.class_of(AbstractCommand))) }
      def command(name) = subclasses.find { _1.command_name == name }

      private

      sig { params(block: T.nilable(T.proc.bind(CLI::Parser).void)).void }
      def cmd_args(&block)
        @parser_block = T.let(block, T.nilable(T.proc.void))
      end
    end

    # @note because `Args` makes use `OpenStruct`, subclasses may need to use a tapioca compiler,
    #   hash accessors, args.rbi, or other means to make this work with legacy commands:
    sig { returns(CLI::Args) }
    attr_reader :args

    sig { params(argv: T::Array[String]).void }
    def initialize(argv = ARGV.freeze)
      @args = T.let(CLI::Parser.new(&self.class.parser_block).parse(argv), CLI::Args)
    end

    sig { abstract.void }
    def run; end
  end
end
