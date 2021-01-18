# typed: strict
# frozen_string_literal: true

module Utils
  extend T::Sig

  sig {
    params(
      repo:   T.any(String, Pathname),
      length: T.nilable(Integer),
      safe:   T::Boolean,
    ).returns(T.nilable(String))
  }
  def self.git_head(repo = Pathname.pwd, length: nil, safe: true)
    return git_short_head(repo, length: length) if length.present?

    repo = Pathname(repo).extend(GitRepositoryExtension)
    repo.git_head(safe: safe)
  end

  sig {
    params(
      repo:   T.any(String, Pathname),
      length: T.nilable(Integer),
      safe:   T::Boolean,
    ).returns(T.nilable(String))
  }
  def self.git_short_head(repo = Pathname.pwd, length: nil, safe: true)
    repo = Pathname(repo).extend(GitRepositoryExtension)
    repo.git_short_head(length: length, safe: safe)
  end
end
