# typed: strict

# This file contains temporary definitions for fixes that have
# been submitted upstream to https://github.com/sorbet/sorbet.

class IO
  # https://github.com/sorbet/sorbet/pull/3722
  sig do
    type_parameters(:U).params(
      fd: T.any(String, Integer),
      mode: T.any(Integer, String),
      opt: T.nilable(T::Hash[Symbol, T.untyped]),
      blk: T.proc.params(io: T.attached_class).returns(T.type_parameter(:U))
    ).returns(T.type_parameter(:U))
  end
  def self.open(fd, mode='r', opt=nil, &blk); end
end

class Pathname
  # https://github.com/sorbet/sorbet/pull/3729
  sig do
    params(
        owner: T.nilable(Integer),
        group: T.nilable(Integer),
    )
    .returns(Integer)
  end
  def chown(owner, group); end
end

module FileUtils
  # https://github.com/sorbet/sorbet/pull/3730
  module_function

  sig do
    params(
      src: T.untyped,
      dest: T.untyped,
      preserve: T.nilable(T::Boolean),
      noop: T.nilable(T::Boolean),
      verbose: T.nilable(T::Boolean)
    ).returns(T.untyped)
  end
  def cp(src, dest, preserve: nil, noop: nil, verbose: nil); end

  sig do
    params(
      list: T.any(String, Pathname),
      mode: T.nilable(Integer),
      noop: T.nilable(T::Boolean),
      verbose: T.nilable(T::Boolean)
    ).returns(T::Array[String])
  end
  def mkdir_p(list, mode: nil, noop: nil, verbose: nil); end
end

class Module
  # https://github.com/sorbet/sorbet/pull/3732
  sig do
    params(
        arg0: T.any(Symbol, String),
        arg1: T.any(Proc, Method, UnboundMethod)
    )
    .returns(Symbol)
  end
  sig do
    params(
        arg0: T.any(Symbol, String),
        blk: T.proc.bind(T.untyped).returns(T.untyped),
    )
    .returns(Symbol)
  end
  def define_method(arg0, arg1=T.unsafe(nil), &blk); end
end
