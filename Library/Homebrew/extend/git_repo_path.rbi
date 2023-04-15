# typed: strict

class GitRepoPath < SimpleDelegator
  include Kernel

  # This is a workaround to enable `alias pathname __getobj__`
  # @see https://github.com/sorbet/sorbet/issues/2378#issuecomment-569474238
  sig { returns(Pathname) }
  def __getobj__; end

  def /(_arg0); end
  def parent; end
  def abv; end
  def rmtree; end
  def cd; end
  def directory?; end
end
