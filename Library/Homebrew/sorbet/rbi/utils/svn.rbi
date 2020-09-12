# typed: strict

module Utils::Svn
  include Kernel

  sig { returns(T::Boolean) }
  def available?; end

  sig { returns(T.nilable(String)) }
  def version; end

  sig { params(url: String).returns(T::Boolean) }
  def remote_exists?(url); end
end
