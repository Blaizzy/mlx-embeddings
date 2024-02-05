# typed: strict

# This file contains temporary definitions for fixes that have
# been submitted upstream to https://github.com/sorbet/sorbet.

# https://github.com/sorbet/sorbet/pull/7650
class Etc::Group < Struct
  sig { returns(Integer) }
  def gid; end
  sig { returns(T::Array[String]) }
  def mem; end
  sig { returns(String) }
  def name; end
  sig { returns(String) }
  def passwd; end
end

# https://github.com/sorbet/sorbet/pull/7647
module IRB
  sig { params(ap_path: T.nilable(String), argv: T::Array[String]).void }
  def self.setup(ap_path, argv: ::ARGV); end
end

# https://github.com/sorbet/sorbet/pull/7678
class String
  sig do
    params(
        arg0: Integer,
        arg1: Integer,
    )
    .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: T.any(T::Range[Integer], Regexp),
    )
    .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: Regexp,
        arg1: Integer,
    )
    .returns(T.nilable(String))
  end
   sig do
     params(
         arg0: Regexp,
         arg1: T.any(String, Symbol),
     )
     .returns(T.nilable(String))
   end
  sig do
    params(
        arg0: String,
    )
    .returns(T.nilable(String))
  end
  def [](arg0, arg1=T.unsafe(nil)); end

  sig do
     params(
         arg0: Integer,
         arg1: Integer,
     )
     .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: T.any(T::Range[Integer], Regexp),
    )
    .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: Regexp,
        arg1: Integer,
    )
    .returns(T.nilable(String))
  end
   sig do
     params(
         arg0: Regexp,
         arg1: T.any(String, Symbol),
     )
     .returns(T.nilable(String))
   end
  sig do
    params(
        arg0: String,
    )
    .returns(T.nilable(String))
  end
  def slice!(arg0, arg1=T.unsafe(nil)); end

  sig do
    params(
        arg0: Integer,
        arg1: Integer,
    )
    .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: T.any(T::Range[Integer], Regexp),
    )
    .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: Regexp,
        arg1: Integer,
    )
    .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: Regexp,
        arg1: T.any(String, Symbol),
    )
    .returns(T.nilable(String))
  end
  sig do
    params(
        arg0: String,
    )
    .returns(T.nilable(String))
  end
  def slice(arg0, arg1=T.unsafe(nil)); end
end
