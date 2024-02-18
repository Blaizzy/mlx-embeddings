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
