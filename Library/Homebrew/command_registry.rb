# typed: strong
# frozen_string_literal: true

module Homebrew
  module CommandRegistry
    extend T::Helpers

    Cmd = T.type_alias { T.class_of(AbstractCommand) } # rubocop:disable Style/MutableConstant

    sig { params(subclass: Cmd).void }
    def self.register(subclass)
      @cmds ||= T.let({}, T.nilable(T::Hash[String, Cmd]))
      @cmds[subclass.command_name] = subclass
    end

    sig { params(name: String).returns(T.nilable(Cmd)) }
    def self.command(name) = @cmds&.[](name)
  end
end
