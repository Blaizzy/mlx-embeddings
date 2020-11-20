# typed: strict

class Pathname
  # https://github.com/sorbet/sorbet/pull/3676
  sig { params(p1: T.any(String, Pathname), p2: String).returns(T::Array[Pathname]) }
  def self.glob(p1, p2 = T.unsafe(nil)); end

  # https://github.com/sorbet/sorbet/pull/3678
  sig { params(with_directory: T::Boolean).returns(T::Array[Pathname]) }
  def children(with_directory = true); end
end
