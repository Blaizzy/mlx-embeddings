# typed: strict

class Time
  sig { returns(T.any(Integer, Float)) }
  def remaining; end

  sig { returns(T.any(Integer, Float)) }
  def remaining!; end
end
