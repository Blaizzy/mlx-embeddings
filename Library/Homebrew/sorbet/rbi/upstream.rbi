# typed: strict

class Pathname
  # https://github.com/sorbet/sorbet/pull/3676
  sig { params(p1: T.any(String, Pathname), p2: String).returns(T::Array[Pathname]) }
  def self.glob(p1, p2 = T.unsafe(nil)); end
end
