# typed: strict

module UnpackStrategy
  include Kernel
end

class Pathname
  sig { returns(String) }
  def magic_number; end

  sig { returns(String) }
  def file_type; end

  sig { returns(T::Array[String]) }
  def zipinfo; end
end
