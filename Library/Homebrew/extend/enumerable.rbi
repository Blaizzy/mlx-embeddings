# typed: strict

module Enumerable
  requires_ancestor { Object }

  sig {
    type_parameters(:key).params(
      block: T.nilable(T.proc.params(o: Enumerable::Elem).returns(T.type_parameter(:key))),
    ).returns(T::Hash[T.type_parameter(:key), Enumerable::Elem])
  }
  def index_by(&block); end
end

class Hash
  sig { returns(T::Hash[Hash::K, Hash::V]) }
  def compact_blank; end
end
