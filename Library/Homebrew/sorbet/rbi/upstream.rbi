# typed: strict

# This file contains temporary definitions for fixes that have
# been submitted upstream to https://github.com/sorbet/sorbet.

module Kernel
  # @see https://github.com/sorbet/sorbet/blob/a1e8389/rbi/core/kernel.rbi#L41-L46
  sig do
    type_parameters(:U).params(
      block: T.proc.params(cont: Continuation).returns(T.type_parameter(:U))
    ).returns(T.type_parameter(:U))
  end
  def callcc(&block); end

  # @see https://github.com/sorbet/sorbet/blob/a1e8389/rbi/core/kernel.rbi#L2348-L2363
  sig do
    params(
      arg0: T.nilable(
        T.proc.params(
          event: String,
          file: String,
          line: Integer,
          id: T.nilable(Symbol),
          binding: T.nilable(Binding),
          classname: Object,
        ).returns(T.untyped)
      )
    ).void
  end
  sig { params(arg0: NilClass).returns(NilClass) }
  def set_trace_func(arg0); end
end
