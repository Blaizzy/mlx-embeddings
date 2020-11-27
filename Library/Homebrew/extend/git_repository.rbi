# typed: strict

module GitRepositoryExtension
  include Kernel

  sig { params(args: T.any(String, Pathname)).returns(Pathname) }
  def join(*args); end
end
