# typed: strict

class Hash
  sig { params(valid_keys: T.untyped).void }
  def assert_valid_keys!(*valid_keys); end
end
