# typed: strict
# frozen_string_literal: true

module Utils
  extend T::Sig

  # Gets the full commit hash of the HEAD commit.
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

  # Gets a short commit hash of the HEAD commit.
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

  # Gets the name of the currently checked-out branch, or HEAD if the repository is in a detached HEAD state.
  sig {
    params(
      repo: T.any(String, Pathname),
      safe: T::Boolean,
    ).returns(T.nilable(String))
  }
  def self.git_branch(repo = Pathname.pwd, safe: true)
    repo = Pathname(repo).extend(GitRepositoryExtension)
    repo.git_branch(safe: safe)
  end

  # Gets the full commit message of the specified commit, or of the HEAD commit if unspecified.
  sig {
    params(
      repo:   T.any(String, Pathname),
      commit: String,
      safe:   T::Boolean,
    ).returns(T.nilable(String))
  }
  def self.git_commit_message(repo = Pathname.pwd, commit: "HEAD", safe: true)
    repo = Pathname(repo).extend(GitRepositoryExtension)
    repo.git_commit_message(commit, safe: safe)
  end
end
