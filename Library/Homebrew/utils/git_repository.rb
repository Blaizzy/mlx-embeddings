# typed: strict
# frozen_string_literal: true

module Utils
  extend T::Sig

  sig { params(repo: T.any(String, Pathname), length: T.nilable(Integer)).returns(T.nilable(String)) }
  def self.git_head(repo, length: nil)
    return git_short_head(repo, length: length) if length.present?

    repo = Pathname(repo).extend(GitRepositoryExtension)
    repo.git_head
  end

  sig { params(repo: T.any(String, Pathname), length: T.nilable(Integer)).returns(T.nilable(String)) }
  def self.git_short_head(repo, length: nil)
    repo = Pathname(repo).extend(GitRepositoryExtension)
    repo.git_short_head(length: length)
  end
end
