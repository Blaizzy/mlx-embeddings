# typed: strict
# frozen_string_literal: true

class MacOSRunnerSpec < T::Struct
  extend T::Sig

  const :name, String
  const :runner, String
  const :cleanup, T::Boolean

  sig { returns({ name: String, runner: String, cleanup: T::Boolean }) }
  def to_h
    {
      name:    name,
      runner:  runner,
      cleanup: cleanup,
    }
  end
end
