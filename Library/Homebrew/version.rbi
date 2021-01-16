# typed: strict

class Version
  # For `alias eql? ==`
  # See discussions:
  #  - https://github.com/sorbet/sorbet/pull/1443
  #  - https://github.com/sorbet/sorbet/issues/2378
  sig { params(other: T.untyped).returns(T::Boolean) }
  def ==(other); end
end
