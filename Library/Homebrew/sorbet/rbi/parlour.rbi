# typed: strict
module Homebrew
  class Cleanup
    sig { returns(T::Boolean) }
    def dry_run?; end

    sig { returns(T::Boolean) }
    def scrub?; end

    sig { returns(T::Boolean) }
    def prune?; end
  end
end

module Debrew
  sig { returns(T::Boolean) }
  def self.active?; end
end

class Formula
  sig { params(arg: T.untyped).returns(T.untyped) }
  def self.desc(arg = T.unsafe(nil)); end

  sig { params(arg: T.untyped).returns(T.untyped) }
  def self.homepage(arg = T.unsafe(nil)); end

  sig { params(arg: T.untyped).returns(T.untyped) }
  def self.revision(arg = T.unsafe(nil)); end

  sig { params(arg: T.untyped).returns(T.untyped) }
  def self.version_scheme(arg = T.unsafe(nil)); end
end

class FormulaInstaller
  sig { returns(T::Boolean) }
  def installed_as_dependency?; end

  sig { returns(T::Boolean) }
  def installed_on_request?; end

  sig { returns(T::Boolean) }
  def show_summary_heading?; end

  sig { returns(T::Boolean) }
  def show_header?; end

  sig { returns(T::Boolean) }
  def force_bottle?; end

  sig { returns(T::Boolean) }
  def ignore_deps?; end

  sig { returns(T::Boolean) }
  def only_deps?; end

  sig { returns(T::Boolean) }
  def interactive?; end

  sig { returns(T::Boolean) }
  def git?; end

  sig { returns(T::Boolean) }
  def force?; end

  sig { returns(T::Boolean) }
  def overwrite?; end

  sig { returns(T::Boolean) }
  def keep_tmp?; end

  sig { returns(T::Boolean) }
  def verbose?; end

  sig { returns(T::Boolean) }
  def debug?; end

  sig { returns(T::Boolean) }
  def quiet?; end

  sig { returns(T::Boolean) }
  def hold_locks?; end
end

class Requirement
  sig { params(arg: T.untyped).returns(T.untyped) }
  def self.fatal(arg = T.unsafe(nil)); end

  sig { params(arg: T.untyped).returns(T.untyped) }
  def self.cask(arg = T.unsafe(nil)); end

  sig { params(arg: T.untyped).returns(T.untyped) }
  def self.download(arg = T.unsafe(nil)); end
end

class BottleSpecification
  sig { params(arg: T.untyped).returns(T.untyped) }
  def rebuild(arg = T.unsafe(nil)); end
end

class SystemCommand
  sig { returns(T::Boolean) }
  def sudo?; end

  sig { returns(T::Boolean) }
  def print_stdout?; end

  sig { returns(T::Boolean) }
  def print_stderr?; end

  sig { returns(T::Boolean) }
  def must_succeed?; end
end

module Cask
  class Audit
    sig { returns(T::Boolean) }
    def appcast?; end

    sig { returns(T::Boolean) }
    def new_cask?; end

    sig { returns(T::Boolean) }
    def strict?; end

    sig { returns(T::Boolean) }
    def online?; end

    sig { returns(T::Boolean) }
    def token_conflicts?; end
  end

  class DSL
    class Caveats < Base
      sig { returns(T::Boolean) }
      def discontinued?; end
    end
  end

  class Installer
    sig { returns(T::Boolean) }
    def binaries?; end

    sig { returns(T::Boolean) }
    def force?; end

    sig { returns(T::Boolean) }
    def skip_cask_deps?; end

    sig { returns(T::Boolean) }
    def require_sha?; end

    sig { returns(T::Boolean) }
    def reinstall?; end

    sig { returns(T::Boolean) }
    def upgrade?; end

    sig { returns(T::Boolean) }
    def verbose?; end

    sig { returns(T::Boolean) }
    def installed_as_dependency?; end

    sig { returns(T::Boolean) }
    def quarantine?; end

    sig { returns(T::Boolean) }
    def quiet?; end
  end
end
