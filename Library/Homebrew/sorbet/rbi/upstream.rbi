# typed: strict

class Pathname
  # https://github.com/sorbet/sorbet/pull/3676
  sig { params(p1: T.any(String, Pathname), p2: String).returns(T::Array[Pathname]) }
  def self.glob(p1, p2 = T.unsafe(nil)); end

  # https://github.com/sorbet/sorbet/pull/3678
  sig { params(with_directory: T::Boolean).returns(T::Array[Pathname]) }
  def children(with_directory = true); end
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
