# typed: strict
# frozen_string_literal: true

require "utils/git"
require "utils/popen"

# Extensions to {Pathname} for querying Git repository information.
# @see Utils::Git
# @api private
module GitRepositoryExtension
  extend T::Sig

  sig { returns(T::Boolean) }
  def git?
    join(".git").exist?
  end

  # Gets the URL of the Git origin remote.
  sig { returns(T.nilable(String)) }
  def git_origin
    return if !git? || !Utils::Git.available?

    Utils.popen_read(Utils::Git.git, "config", "--get", "remote.origin.url", chdir: self).chomp.presence
  end

  # Sets the URL of the Git origin remote.
  sig { params(origin: String).returns(T.nilable(T::Boolean)) }
  def git_origin=(origin)
    return if !git? || !Utils::Git.available?

    safe_system Utils::Git.git, "remote", "set-url", "origin", origin, chdir: self
  end

  # Gets the full commit hash of the HEAD commit.
  sig { params(safe: T::Boolean).returns(T.nilable(String)) }
  def git_head(safe: false)
    popen_git("rev-parse", "--verify", "-q", "HEAD", safe: safe)
  end

  # Gets a short commit hash of the HEAD commit.
  sig { params(length: T.nilable(Integer), safe: T::Boolean).returns(T.nilable(String)) }
  def git_short_head(length: nil, safe: false)
    short_arg = length.present? ? "--short=#{length}" : "--short"
    popen_git("rev-parse", short_arg, "--verify", "-q", "HEAD", safe: safe)
  end

  # Gets the relative date of the last commit, e.g. "1 hour ago"
  sig { returns(T.nilable(String)) }
  def git_last_commit
    return if !git? || !Utils::Git.available?

    Utils.popen_read(Utils::Git.git, "show", "-s", "--format=%cr", "HEAD", chdir: self).chomp.presence
  end

  # Gets the name of the currently checked-out branch, or HEAD if the repository is in a detached HEAD state.
  sig { returns(T.nilable(String)) }
  def git_branch
    return if !git? || !Utils::Git.available?

    Utils.popen_read(Utils::Git.git, "rev-parse", "--abbrev-ref", "HEAD", chdir: self).chomp.presence
  end

  # Gets the name of the default origin HEAD branch.
  sig { returns(T.nilable(String)) }
  def git_origin_branch
    return if !git? || !Utils::Git.available?

    Utils.popen_read(Utils::Git.git, "symbolic-ref", "-q", "--short", "refs/remotes/origin/HEAD", chdir: self)
         .chomp.presence&.split("/")&.last
  end

  # Returns true if the repository's current branch matches the default origin branch.
  sig { returns(T.nilable(T::Boolean)) }
  def git_default_origin_branch?
    git_origin_branch == git_branch
  end

  # Returns the date of the last commit, in YYYY-MM-DD format.
  sig { returns(T.nilable(String)) }
  def git_last_commit_date
    return if !git? || !Utils::Git.available?

    Utils.popen_read(Utils::Git.git, "show", "-s", "--format=%cd", "--date=short", "HEAD", chdir: self).chomp.presence
  end

  # Gets the full commit message of the specified commit, or of the HEAD commit if unspecified.
  sig { params(commit: String).returns(T.nilable(String)) }
  def git_commit_message(commit = "HEAD")
    return if !git? || !Utils::Git.available?

    Utils.popen_read(Utils::Git.git, "log", "-1", "--pretty=%B", commit, "--", chdir: self, err: :out).strip.presence
  end

  private

  sig { params(args: T.untyped, safe: T::Boolean).returns(T.nilable(String)) }
  def popen_git(*args, safe: false)
    unless git?
      return unless safe

      raise "Not a Git repository: #{self}"
    end

    unless Utils::Git.available?
      return unless safe

      raise "Git is unavailable"
    end

    T.unsafe(Utils).popen_read(Utils::Git.git, *args, safe: safe, chdir: self).chomp.presence
  end
end
