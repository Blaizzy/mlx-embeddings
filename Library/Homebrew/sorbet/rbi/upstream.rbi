# typed: strict

# This file contains temporary definitions for fixes that have
# been submitted upstream to https://github.com/sorbet/sorbet.

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
