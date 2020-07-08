# typed: strict

module Git
  include Kernel

  def last_revision_commit_of_file(repo, file, before_commit: nil)
  end

  sig { params(repo: Pathname, files: T::Array[Pathname], before_commit: T.nilable(String)).void }
  def last_revision_commit_of_files(repo, files, before_commit: nil)
  end

  def last_revision_of_file(repo, file, before_commit: nil)
  end
end
