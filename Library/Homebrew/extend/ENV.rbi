# typed: strict

# @!visibility private
module EnvMethods
  include Kernel

  sig { params(key: String).returns(T::Boolean) }
  def key?(key); end

  sig { params(key: String).returns(T.nilable(String)) }
  def [](key); end

  sig { params(key: String).returns(String) }
  def fetch(key); end

  sig { params(key: String, value: T.nilable(T.any(String, PATH))).returns(T.nilable(String)) }
  def []=(key, value); end

  sig { params(block: T.proc.params(arg0: [String, String]).returns(T::Boolean)).returns(T::Hash[String, String]) }
  def select(&block); end

  sig { params(block: T.proc.params(arg0: String).void).void }
  def each_key(&block); end

  sig { params(key: String).returns(T.nilable(String)) }
  def delete(key); end

  sig {
    params(other: T.any(T::Hash[String, String], Sorbet::Private::Static::ENVClass))
      .returns(Sorbet::Private::Static::ENVClass)
  }
  def replace(other); end

  sig { returns(T::Hash[String, String]) }
  def to_hash; end
end

module EnvActivation
  include EnvMethods
  include Superenv
end

# @!visibility private
class Sorbet
  module Private
    module Static
      class ENVClass
        include EnvActivation
      end
    end
  end
end
