# typed: strong
# frozen_string_literal: true

module Homebrew
  # Subclass this to implement a `brew` command. This is preferred to declaring a named function in the `Homebrew`
  # module, because:
  # - Each Command lives in an isolated namespace.
  # - Each Command implements a defined interface.
  # - `args` is available as an ivar, and thus does not need to be passed as an argument to helper methods.
  #
  # To subclass, implement a `run` method and provide a `cmd_args` block to document the command and its allowed args.
  # To generate method signatures for command args, run `brew typecheck --update`.
  class AbstractCommand
    extend T::Helpers

    abstract!

    class << self
      sig { returns(T.nilable(CLI::Parser)) }
      attr_reader :parser

      sig { returns(String) }
      def command_name = T.must(name).split("::").fetch(-1).downcase

      # @return the AbstractCommand subclass associated with the brew CLI command name.
      sig { params(name: String).returns(T.nilable(T.class_of(AbstractCommand))) }
      def command(name) = subclasses.find { _1.command_name == name }

      private

      sig { params(block: T.proc.bind(CLI::Parser).void).void }
      def cmd_args(&block)
        @parser = T.let(CLI::Parser.new(&block), T.nilable(CLI::Parser))
      end
    end

    sig { returns(CLI::Args) }
    attr_reader :args

    sig { params(argv: T::Array[String]).void }
    def initialize(argv = ARGV.freeze)
      parser = self.class.parser
      raise "Commands must include a `cmd_args` block" if parser.nil?

      @args = T.let(parser.parse(argv), CLI::Args)
    end

    sig { abstract.void }
    def run; end
  end
end
