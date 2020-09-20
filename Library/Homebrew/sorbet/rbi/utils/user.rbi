# typed: strict

class User < SimpleDelegator
  include Kernel

  sig { returns(T::Boolean) }
  def gui?; end

  sig { returns(T.nilable(T.attached_class)) }
  def self.current; end
end
