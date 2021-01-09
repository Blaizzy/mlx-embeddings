# typed: true
# frozen_string_literal: true

require "fcntl"

# A lock file.
#
# @api private
class LockFile
  attr_reader :path

  def initialize(name)
    @name = name.to_s
    @path = HOMEBREW_LOCKS/"#{@name}.lock"
    @lockfile = nil
  end

  def lock
    @path.parent.mkpath
    create_lockfile
    return if @lockfile.flock(File::LOCK_EX | File::LOCK_NB)

    raise OperationInProgressError, @name
  end

  def unlock
    return if @lockfile.nil? || @lockfile.closed?

    @lockfile.flock(File::LOCK_UN)
    @lockfile.close
  end

  def with_lock
    lock
    yield
  ensure
    unlock
  end

  private

  def create_lockfile
    return if @lockfile.present? && !@lockfile.closed?

    @lockfile = @path.open(File::RDWR | File::CREAT)
    @lockfile.fcntl(Fcntl::F_SETFD, Fcntl::FD_CLOEXEC)
  end
end

# A lock file for a formula.
#
# @api private
class FormulaLock < LockFile
  def initialize(name)
    super("#{name}.formula")
  end
end

# A lock file for a cask.
#
# @api private
class CaskLock < LockFile
  def initialize(name)
    super("#{name}.cask")
  end
end
