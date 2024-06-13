# typed: strict

# This file contains temporary definitions for fixes that have
# been submitted upstream to https://github.com/sorbet/sorbet.

# https://github.com/sorbet/sorbet/pull/7959
module FileUtils
  sig {
    params(
      src:                T.any(File, String, Pathname, T::Array[T.any(File, String, Pathname)]),
      dest:               T.any(String, Pathname),
      preserve:           T.nilable(T::Boolean),
      noop:               T.nilable(T::Boolean),
      verbose:            T.nilable(T::Boolean),
      dereference_root:   T::Boolean,
      remove_destination: T.nilable(T::Boolean),
    ).returns(T.nilable(T::Array[String]))
  }
  def self.cp_r(src, dest, preserve: nil, noop: nil, verbose: nil, dereference_root: true, remove_destination: nil)
    # XXX: This comment is a placeholder to suppress `Style/EmptyMethod` lint.
    # Simply compacting the method definition in a single line would in turn trigger
    # `Layout/LineLength`, driving `brew style --fix` to an infinite loop.
  end
end
