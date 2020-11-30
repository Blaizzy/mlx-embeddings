# typed: strict

module DiskUsageExtension
  include Kernel

  def exist?; end

  def symlink?; end

  def resolved_path; end
end

module ObserverPathnameExtension
  include Kernel

  def dirname; end

  def basename; end
end
