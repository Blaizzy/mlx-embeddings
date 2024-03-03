# typed: strong
# frozen_string_literal: true

module Homebrew
  class AbstractCommand
    extend T::Helpers

    abstract!

    class << self
      # registers subclasses for lookup by command name
      sig { params(subclass: T.class_of(AbstractCommand)).void }
      def inherited(subclass)
        super
        @cmds ||= T.let({}, T.nilable(T::Hash[String, T.class_of(AbstractCommand)]))
        @cmds[subclass.command_name] = subclass
      end

      sig { params(name: String).returns(T.nilable(T.class_of(AbstractCommand))) }
      def command(name) = @cmds&.[](name)

      sig { returns(String) }
      def command_name = T.must(name).split("::").fetch(-1).downcase
    end

    # @note because `Args` makes use `OpenStruct`, subclasses may need to use a tapioca compiler,
    #   hash accessors, args.rbi, or other means to make this work with legacy commands:
    sig { returns(Homebrew::CLI::Args) }
    attr_reader :args

    sig { void }
    def initialize
      @args = T.let(raw_args.parse, Homebrew::CLI::Args)
    end

    sig { abstract.returns(CLI::Parser) }
    def raw_args; end

    sig { abstract.void }
    def run; end
  end
end
